

run:
    , .venv/bin/activate

    # ablate
    uv python nbs/train_svft.py --quick --use_wandb
