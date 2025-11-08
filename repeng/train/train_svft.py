#!/usr/bin/env python3
"""Train contrastive SVFT adapter for steering LLMs.

Example usage:
    python -m repeng.train.train_svft --batch_size 14 --n_epochs 30
    python -m repeng.train.train_svft --quick --use_wandb
"""

import gc
import json
import random
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Literal, Optional

import pandas as pd
import torch
import tyro
from baukit.nethook import TraceDict
from datasets import Dataset
from loguru import logger
from peft import PeftModel
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorWithPadding,
    GenerationConfig,
)

from repeng import ControlVector, make_dataset
from repeng.adapter import ScaleAdapter
from repeng.extract import _collect_activations_only, read_representations
from repeng.peft_utils.svft import TRMSvftAConfig, TRMSvftModel
from repeng.train.daily_dilemas import (
    evaluate_daily_dilemma,
    load_and_process_dataset,
    load_labels,
    process_daily_dilemma_results,
    select_dilemma_by_values,
)
from repeng.train.inner_contrastive_loss import contrastive_steering_loss_with_ref

proj_root = Path(__file__).parent.parent.parent


@dataclass
class TrainingConfig:
    """Configuration for training contrastive SVFT adapter."""

    # Model config
    model_name: str = "Qwen/Qwen3-4B-Instruct-2507"
    quantization_type: Literal["4bit", "8bit", "none"] = "none"
    target_modules: str = ".*\.(5|7|10|13|15|20|25|30|33)\..*(gate_proj|down_proj)"

    # Training params
    batch_size: int = 14
    n_epochs: int = 30
    lr: float = 1e-3
    weight_decay: float = 0.01
    log_n: int = 40
    grad_accum_steps: int = 6
    quick: bool = False

    # Adapter params
    rank: int = 64
    scale_s: Literal["add", "mult", "none"] = "mult"
    svft_rotate_u: bool = True
    svft_rotate_v: bool = True
    full_loss_u: bool = True

    # Dataset
    dataset_name: str = "honest"
    dataset_max_size: int = 128

    # Loss
    use_logsigmoid: bool = True
    coherence_threshold: float = 1.5
    boundary_order: int = 1
    last_n_tokens: int = 6

    # Eval
    eval_batch_size: Optional[int] = None
    eval_dilemmas: int = 48

    # Output
    output_dir: str = "../outputs/adapters"
    use_wandb: bool = False
    wandb_project: str = "repeng-steering"
    save_checkpoints: bool = False


def setup_logging():
    """Configure loguru for clean output."""
    logger.remove()
    logger.add(
        lambda msg: tqdm.write(msg, end=""),
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
        colorize=True,
    )


def clear_mem():
    """Clear GPU memory."""
    gc.collect()
    torch.cuda.empty_cache()


def register_svft_peft():
    """Register custom SVFT adapter with PEFT."""
    import enum

    import peft.utils.peft_types
    from peft.mapping import PEFT_TYPE_TO_PREFIX_MAPPING
    from peft.utils import register_peft_method

    class PeftType2(str, enum.Enum):
        TRMSVFT = "TRMSVFT"

    peft.utils.peft_types.PeftType = PeftType2
    PEFT_TYPE_TO_PREFIX_MAPPING[TRMSvftAConfig.peft_type] = "svft_"
    register_peft_method(
        name="trmsvft",
        model_cls=TRMSvftModel,
        config_cls=TRMSvftAConfig,
        prefix="svft_",
    )


def load_suffixes(data_dir: Path = proj_root / "notebooks/data") -> List[str]:
    """Load dataset suffixes from JSON files."""
    random.seed(42)
    suffix_files = data_dir.glob("*.json")
    suffixes = []

    for sf in suffix_files:
        with open(sf) as f:
            f_suffixes = json.load(f)
            random.shuffle(f_suffixes)
            suffixes += f_suffixes[:128]

    logger.info(f"Loaded {len(suffixes)} suffixes from {data_dir}")
    return suffixes


def create_dataset(config: TrainingConfig, tokenizer):
    """Create contrastive dataset."""
    suffixes = load_suffixes()

    honest_dataset = make_dataset(
        "Pretend you're an {persona} person making statements about the world.",
        ["honest", "truthful", "ethical", "moral"],
        ["dishonest", "deceptive", "untruthful", "immoral"],
        suffixes,
        tokenizer,
    )

    data = []
    for ex in honest_dataset:
        data.append({"s": ex.positive})
        data.append({"s": ex.negative})

    dataset = Dataset.from_list(data)

    if config.quick:
        dataset = dataset.select(range(256))
        honest_dataset = honest_dataset[:256]

    logger.info(
        f"Dataset: {len(dataset)} examples, {len(honest_dataset)} contrastive pairs"
    )

    # Tokenize
    dataset_pt = dataset.map(
        lambda examples: tokenizer(examples["s"], truncation=True, max_length=512),
        batched=True,
        remove_columns=["s"],
    )
    dataset_pt.set_format(type="torch", columns=["input_ids", "attention_mask"])

    return honest_dataset, dataset_pt


def load_model(config: TrainingConfig):
    """Load base model with optional quantization."""
    quantization_config = None
    if config.quantization_type == "4bit":
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=False,
            bnb_4bit_quant_type="nf4",
        )
    elif config.quantization_type == "8bit":
        quantization_config = BitsAndBytesConfig(load_in_8bit=True)

    logger.info(f"Loading model: {config.model_name}")
    base_model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float16,
        quantization_config=quantization_config,
        device_map="cuda:0",
    )

    if quantization_config is not None:
        base_model.enable_input_require_grads()

    return base_model


def setup_adapter(base_model, config: TrainingConfig):
    """Setup SVFT adapter on base model."""
    adapter_config = TRMSvftAConfig(
        r=config.rank,
        scale_s=config.scale_s,
        rotate_u=config.svft_rotate_u,
        rotate_v=config.svft_rotate_v,
        task_type="CAUSAL_LM",
        target_modules=config.target_modules,
    )

    model = PeftModel(base_model, adapter_config, adapter_name=config.dataset_name)
    logger.info(
        f"Adapter configured: rank={config.rank}, target_modules={config.target_modules}"
    )

    return model


def get_loss_layers(model, config: TrainingConfig):
    """Determine which layers to apply loss to."""
    import re

    adapter_layers = [
        name for name, param in model.named_parameters() if param.requires_grad
    ]
    parent_layers = sorted(
        set([".".join(layer.split(".")[:-2]) for layer in adapter_layers])
    )

    model_max_layers = model.config.num_hidden_layers
    suffixes = set([p.split(".")[-1] for p in parent_layers])

    loss_layers = []
    for suffix in suffixes:
        candidates = [p for p in parent_layers if p.endswith(suffix)]

        def get_layer_num(s):
            m = re.search(r"\.layers\.(\d+)\.", s)
            return int(m.group(1)) if m else None

        candidate = None
        candidate_i = 0
        for c in candidates:
            i = get_layer_num(c)
            if i is None:
                continue
            if (candidate is None) or ((i > candidate_i) and i <= model_max_layers - 3):
                candidate = c
                candidate_i = i

        if candidate is not None:
            loss_layers.append(candidate)

    logger.info(f"Loss layers: {loss_layers}")
    return loss_layers


def extract_U_matrices(model, loss_layers: List[str], config: TrainingConfig):
    """Extract SVD U matrices for loss projection."""
    Uw_full = {}

    if config.full_loss_u:
        for lk in loss_layers:
            m = model.get_submodule(lk)
            W = m.weight.data.float()
            U, S, Vh = torch.linalg.svd(W, full_matrices=False)
            Uw_full[lk] = U[:, :-1].to(model.device).float()

        shapes = {k: v.shape for k, v in Uw_full.items()}
        logger.info(f"Extracted U matrices: {shapes}")

    return Uw_full


def train_steer_vector(model, honest_dataset, loss_layers, Uw_full, tokenizer, config):
    """Extract steering directions in U-space."""
    model.eval()

    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
        train_strs = [s for ex in honest_dataset for s in (ex.positive, ex.negative)]
        last_act, logprobs = _collect_activations_only(
            model, tokenizer, train_strs, loss_layers, batch_size=2
        )

    steer_dirs = {}
    for layer in loss_layers:
        U_T = Uw_full[layer]
        hs_cpu = last_act[layer].float()
        hs_proj = hs_cpu @ U_T.cpu()

        h_cho = hs_proj[::2]
        h_rej = hs_proj[1::2]
        steer_dir = (h_cho - h_rej).mean(dim=0)
        steer_dirs[layer] = torch.nn.functional.normalize(steer_dir, dim=0)

    dirs_Uw = ControlVector(model_type=model.config.model_type, directions=steer_dirs)
    dirs_pca = read_representations(last_act, logprobs, grads=None)
    dirs_pca = ControlVector(model_type=model.config.model_type, directions=dirs_pca)

    logger.info("Extracted steering vectors (U-space and PCA)")
    return dirs_Uw, dirs_pca


def get_choice_ids(tokenizer):
    """Get token IDs for Yes/No choices."""

    def is_choice(choice: str, match: str) -> bool:
        return (
            match.lower().endswith(choice) or match.lower().startswith(choice)
        ) and len(match) < len(choice) + 2

    positive_choices = {k: v for k, v in tokenizer.vocab.items() if is_choice("yes", k)}
    negative_choices = {k: v for k, v in tokenizer.vocab.items() if is_choice("no", k)}

    logger.debug(
        f"Choice tokens: yes={list(positive_choices.keys())}, no={list(negative_choices.keys())}"
    )
    return [list(negative_choices.values()), list(positive_choices.values())]


def process_infos(infos, by_layer=True, by_coef=True, by_layer_num=True, verbose=False):
    """Process training info logs into summary dataframe."""
    df_infos = pd.DataFrame(infos)
    df_infos["layer_num"] = df_infos["layer"].str.extract(r"\.(\d+)\.").astype(int)

    if verbose and by_layer_num:
        df_layer_num = df_infos.groupby(["layer_num"])["loss_total"].mean()
        logger.debug(f"Loss by layer_num:\n{df_layer_num}")

    if verbose and by_layer:
        df_layer = df_infos.groupby(["layer"])["loss_total"].mean()
        logger.debug(f"Loss by layer:\n{df_layer}")

    if verbose and by_coef:
        df_coef = df_infos.groupby(["coef"])["loss_total"].mean()
        logger.debug(f"Loss by coef:\n{df_coef}")

    agg_dict = {
        col: "mean" if pd.api.types.is_numeric_dtype(dtype) else "first"
        for col, dtype in df_infos.dtypes.items()
    }
    del agg_dict["step"]
    df_hist = df_infos.groupby("step").agg(agg_dict).drop(columns=["layer", "coef"])

    return df_hist


def train_epoch(
    model,
    train_dataloader,
    dirs_Uw,
    loss_layers,
    Uw_full,
    opt,
    scheduler,
    config: TrainingConfig,
    epoch: int,
    infos: List[dict],
    wandb_run=None,
):
    """Train for one epoch."""
    model.train()

    for j, batch in enumerate(
        tqdm(train_dataloader, desc=f"Epoch {epoch}", leave=False)
    ):
        step = epoch * len(train_dataloader) + j
        batch = {k: v.to(model.device) for k, v in batch.items()}

        attention_mask = batch["attention_mask"]
        mask_cho = attention_mask[::2]
        mask_rej = attention_mask[1::2]
        mask = (mask_cho + mask_rej).clamp(0, 1)

        # Reference outputs
        with torch.no_grad(), ScaleAdapter(model, coeff=None):
            with TraceDict(model, layers=loss_layers) as ret_ref:
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    outputs_ref = model(**batch, output_hidden_states=True)

        ref_logp = outputs_ref.logits[:, :-1].log_softmax(-1)
        labels = batch["input_ids"][:, 1:].unsqueeze(-1)
        ref_label_logp = ref_logp.gather(2, labels).squeeze(-1).float()
        ref_cho_label_logp = ref_label_logp[::2].detach()
        ref_rej_label_logp = ref_label_logp[1::2].detach()

        total_loss = torch.tensor(0.0, device=model.device)

        # Contrastive training with both coefficients
        for coef in [-1.0, 1.0]:
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                with ScaleAdapter(model, coeff=coef):
                    with TraceDict(model, layers=loss_layers, retain_grad=True) as ret:
                        outputs_pi = model(**batch, output_hidden_states=True)

            for lk in loss_layers:
                hs_ref = (ret_ref[lk].output * attention_mask.unsqueeze(-1)).float()
                pref_dir_ref_dH_Uw = (
                    dirs_Uw.directions[lk].clone().to(model.device).float()
                )

                hs_ref_cho = hs_ref[::2]
                hs_ref_rej = hs_ref[1::2]

                hs_pi = (ret[lk].output * attention_mask.unsqueeze(-1)).float()
                hs_pi_cho = hs_pi[::2]
                hs_pi_rej = hs_pi[1::2]

                pi_logprobs = outputs_pi.logits[:, :-1].log_softmax(-1)
                pi_label_logprobs = pi_logprobs.gather(2, labels).squeeze(-1).float()
                pi_rej_label_logp = pi_label_logprobs[1::2]
                pi_cho_label_logp = pi_label_logprobs[::2]

                U_w = (
                    Uw_full[lk]
                    if config.full_loss_u
                    else model.get_submodule(lk)
                    .svft_u[config.dataset_name]
                    .to(model.device)
                    .float()
                )

                ref_coherence = ref_cho_label_logp if coef > 0 else ref_rej_label_logp
                pi_coherence = pi_cho_label_logp if coef > 0 else pi_rej_label_logp

                loss, info1 = contrastive_steering_loss_with_ref(
                    pref_dir=pref_dir_ref_dH_Uw.detach(),
                    hs_ref_cho=hs_ref_cho @ U_w,
                    hs_ref_rej=hs_ref_rej @ U_w,
                    hs_pi_pos=hs_pi_cho @ U_w,
                    hs_pi_neg=hs_pi_rej @ U_w,
                    ref_pos_label_logp=ref_coherence,
                    pi_pos_label_logp=pi_coherence,
                    cho_mask=mask,
                    coef=coef,
                    coherence_threshold=config.coherence_threshold,
                    boundary_order=config.boundary_order,
                    last_n_tokens=config.last_n_tokens,
                    use_logsigmoid=config.use_logsigmoid,
                )

                total_loss += loss.mean()

                info1["lr"] = torch.tensor(scheduler.get_last_lr()[0])
                info1 = {k: v.mean().detach().cpu().item() for k, v in info1.items()}
                info1["coef"] = coef
                info1["layer"] = lk
                info1["step"] = step
                infos.append(info1)

        total_loss.backward()

        if step % config.grad_accum_steps == 0:
            opt.step()
            scheduler.step()
            opt.zero_grad()
            model.zero_grad()
            clear_mem()

        # Logging
        if (
            step
            % (
                len(train_dataloader)
                * config.n_epochs
                // config.grad_accum_steps
                // config.log_n
            )
            == 0
        ):
            df_hist = process_infos(
                infos, by_layer=False, by_coef=True, by_layer_num=True, verbose=False
            )
            if len(df_hist) > 0:
                info = df_hist.iloc[-1].to_dict()
                log_str = " | ".join([f"{k}={v:.3g}" for k, v in info.items()])
                logger.info(f"Step {step}: {log_str}")

                if wandb_run is not None:
                    wandb_run.log(info, step=step)

        if epoch % 5 == 0 and j == 0:
            clear_mem()


def evaluate_model(model, tokenizer, config: TrainingConfig, dirs_pca):
    """Run evaluation on Daily Dilemmas dataset."""
    logger.info("Running evaluation...")
    model.eval()

    dataset_dd, dataset_dd_pt = load_and_process_dataset(
        tokenizer, max_size=config.dataset_max_size
    )
    dataset_dd = select_dilemma_by_values(
        dataset_dd, label="truth", N=config.eval_dilemmas
    )
    dataset_dd_pt = dataset_dd.select_columns(
        ["dilemma_idx", "idx", "input_ids"]
    ).with_format("torch")
    df_labels = load_labels(dataset_dd)

    choice_ids = get_choice_ids(tokenizer)
    generation_config = GenerationConfig(
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        use_cache=True,
        output_logits=True,
        return_dict_in_generate=True,
    )

    eval_batch_size = config.eval_batch_size or config.batch_size // 4

    df_res = []
    for coeff in [-1.0, 0.0, 1.0]:
        logger.info(f"Evaluating adapter coeff={coeff}")
        clear_mem()
        with ScaleAdapter(model, coeff=coeff):
            d = evaluate_daily_dilemma(
                model,
                dataset_dd_pt,
                tokenizer,
                choice_ids,
                batch_size=eval_batch_size,
                generation_config=generation_config,
            )
            d["coeff"] = coeff
            d["method"] = "train"
            df_res.append(d)

    # Also eval PCA baseline
    from repeng.control import steer

    for coeff in [-1.0, 0.0, 1.0]:
        logger.info(f"Evaluating PCA baseline coeff={coeff}")
        clear_mem()
        with steer(model, dirs_pca, coeff=coeff, retain_grad=False):
            d = evaluate_daily_dilemma(
                model,
                dataset_dd_pt,
                tokenizer,
                choice_ids,
                batch_size=eval_batch_size,
                generation_config=generation_config,
            )
            d["coeff"] = coeff
            d["method"] = "pca"
            df_res.append(d)

    df_res2 = pd.concat(df_res)
    res = process_daily_dilemma_results(df_res2, dataset_dd, df_labels)[0]

    cols_labels = [c for c in res.columns if c.startswith("score_")]
    df_res_pv = res.groupby(["method", "coeff"])[cols_labels].mean().T

    logger.info(f"Evaluation results:\n{df_res_pv}")

    # Key metrics
    for n, g in res.groupby("method"):
        corr_truth = g[["coeff", "score_Virtue/Truthfulness"]].corr().iloc[0, 1]
        corr_logratio = g[["coeff", "logratio"]].corr().iloc[0, 1]
        logger.info(
            f"{n}: truthfulness_corr={corr_truth:.3f}, logratio_corr={corr_logratio:.3f}"
        )

    return res, df_res_pv


def save_adapter(model: PeftModel, save_folder: Path, adapter_name: str):
    """Save adapter weights and config."""
    import safetensors
    from peft.mapping import PEFT_TYPE_TO_PREFIX_MAPPING

    save_folder.mkdir(parents=True, exist_ok=True)

    config = model.peft_config[adapter_name]
    state_dict = model.state_dict()

    prefix = PEFT_TYPE_TO_PREFIX_MAPPING[config.peft_type]
    to_return = {k: state_dict[k] for k in state_dict if prefix in k}

    def remove_adapter_name(key):
        if "." not in key:
            return key
        if key.endswith(f".{adapter_name}"):
            return key.removesuffix(f".{adapter_name}")
        return key.replace(f".{adapter_name}.", ".")

    to_return = {remove_adapter_name(k): v for k, v in to_return.items()}

    safetensors.torch.save_file(to_return, save_folder / "adapter_model.safetensors")
    config.save_pretrained(save_folder)

    logger.info(f"Saved adapter to {save_folder}")


def main(config: TrainingConfig):
    """Main training pipeline."""
    setup_logging()
    logger.info(f"Starting training with config:\n{asdict(config)}")

    # Setup W&B if requested
    wandb_run = None
    if config.use_wandb:
        import wandb

        wandb_run = wandb.init(project=config.wandb_project, config=asdict(config))
        logger.info(f"W&B run: {wandb_run.get_url()}")

    # Register SVFT
    register_svft_peft()

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token_id = 0
    tokenizer.padding_side = "left"

    # Create dataset
    honest_dataset, dataset_pt = create_dataset(config, tokenizer)

    # Load model and adapter
    base_model = load_model(config)
    model = setup_adapter(base_model, config)

    # Setup loss layers
    loss_layers = get_loss_layers(model, config)
    Uw_full = extract_U_matrices(model, loss_layers, config)

    # Extract steering vectors
    with ScaleAdapter(model, coeff=None):
        dirs_Uw, dirs_pca = train_steer_vector(
            model, honest_dataset, loss_layers, Uw_full, tokenizer, config
        )

    logger.info(f"Steering extraction layer: {loss_layers}")

    # Setup training
    data_collator = DataCollatorWithPadding(
        tokenizer=tokenizer, padding="longest", max_length=64
    )
    train_dataloader = DataLoader(
        dataset_pt,
        shuffle=False,
        batch_size=config.batch_size,
        collate_fn=data_collator,
    )

    total_steps = config.n_epochs * len(train_dataloader) // config.grad_accum_steps + 1
    opt = torch.optim.AdamW(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=config.lr, total_steps=total_steps, pct_start=0.3
    )

    logger.info(f"Training: {config.n_epochs} epochs, {total_steps} steps")

    # Training loop
    infos = []
    for epoch in tqdm(range(config.n_epochs), desc="Epochs"):
        train_epoch(
            model,
            train_dataloader,
            dirs_Uw,
            loss_layers,
            Uw_full,
            opt,
            scheduler,
            config,
            epoch,
            infos,
            wandb_run,
        )

    # Process final results
    df_hist = process_infos(infos)
    logger.info(f"Training complete. Final loss: {df_hist['loss_total'].iloc[-1]:.4f}")

    # Evaluation
    res, df_res_pv = evaluate_model(model, tokenizer, config, dirs_pca)

    # Save results
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_folder = (
        Path(config.output_dir) / f"{config.dataset_name}_contrastive_svft_{ts}"
    )
    save_folder.mkdir(parents=True, exist_ok=True)

    save_adapter(model, save_folder, config.dataset_name)

    # Save training config
    with open(save_folder / "training_config.json", "w") as f:
        json.dump(asdict(config), f, indent=4)

    # Save training history
    df_hist.to_csv(save_folder / "training_history.csv", index=False)

    # Save evaluation results
    res.to_csv(save_folder / "eval_results.csv", index=False)
    df_res_pv.to_csv(save_folder / "eval_summary.csv")

    logger.success(f"All results saved to {save_folder}")

    if wandb_run is not None:
        wandb_run.log(
            {
                "eval/truthfulness_corr": res.groupby("method")
                .apply(
                    lambda g: g[["coeff", "score_Virtue/Truthfulness"]]
                    .corr()
                    .iloc[0, 1]
                )
                .to_dict()
            }
        )
        wandb_run.finish()


if __name__ == "__main__":
    config = tyro.cli(TrainingConfig)
    main(config)
