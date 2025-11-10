

run:
    #!/bin/bash -x
    . .venv/bin/activate

    CMD="uv run python nbs/train.py --eval_max_n_dilemmas=64"

    # TODO ablate
    $CMD --no-use-logsigmoid
    $CMD --no-ipissa-rotate-v
    $CMD --no-ipissa-rotate-u
    $CMD --scale-s=none
    $CMD --scale-s=add
    $CMD --no-ipissa-rotate-u --no-ipissa-rotate-v
    $CMD --scale-s=none
    $CMD --weight-decay=0
    $CMD --no-ipissa-rotate-u --no-ipissa-rotate-v  --scale-s=none

    $CMD --lr=1e-2
    $CMD --lr=1e-4 --weight-decay=0.1
    $CMD --rank=8
    $CMD --rank=512
    $CMD --target-modules=".*\.(5|7|10|13|15|20|25|30|33)\..*(gate_proj|down_proj)"
    $CMD --target-modules=".*\.(7|13|20|30|33)\..*(o_proj|up_proj)"
    $CMD --target-modules=".*\.(5|15|25|33)\..*"  --rank=16

    CMD="uv run python nbs/train.py"

    # TODO ablate
    $CMD --no-use-logsigmoid
    $CMD --no-ipissa-rotate-v
    $CMD --no-ipissa-rotate-u
    $CMD --scale-s=none
    $CMD --scale-s=add
    $CMD --no-ipissa-rotate-u --no-ipissa-rotate-v
    $CMD --scale-s=none
    $CMD --weight-decay=0
    $CMD --no-ipissa-rotate-u --no-ipissa-rotate-v  --scale-s=none

    $CMD --lr=1e-2
    $CMD --lr=1e-4 --weight-decay=0.1
    $CMD --lr=1e-4 --weight-decay=1.0
    $CMD --rank=8
    $CMD --rank=512
    $CMD --target-modules=".*\.(5|7|10|13|15|20|25|30|33)\..*(gate_proj|down_proj)"
    $CMD --target-modules=".*\.(7|13|20|30|33)\..*(o_proj|up_proj)"
    $CMD --target-modules=".*\.(5|15|25|33)\..*"  --rank=16
