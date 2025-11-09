

run:
    #!/bin/bash -x
    . .venv/bin/activate

    CMD="uv run python nbs/train.py $*"

    # TODO ablate
    $CMD --no-use-logsigmoid
    uv run python nbs/train.py --no-ipissa-rotate-v
    uv run python nbs/train.py --no-ipissa-rotate-u
    uv run python nbs/train.py --scale-s=none
    uv run python nbs/train.py --scale-s=add
    uv run python nbs/train.py --no-ipissa-rotate-u --no-ipissa-rotate-v
    uv run python nbs/train.py --scale-s=none
    uv run python nbs/train.py --weight-decay=0
    uv run python nbs/train.py --no-ipissa-rotate-u --no-ipissa-rotate-v  --scale-s=none

    uv run python nbs/train.py --lr=1e-2
    uv run python nbs/train.py --lr=1e-4 --weight-decay=0.1
    uv run python nbs/train.py --rank=8
    uv run python nbs/train.py --rank=512
    uv run python nbs/train.py --target-modules=".*\.(5|7|10|13|15|20|25|30|33)\..*(gate_proj|down_proj)"
    uv run python nbs/train.py --target-modules=".*\.(7|13|20|30|33)\..*(o_proj|up_proj)"
    uv run python nbs/train.py --target-modules=".*\.(5|15|25|33)\..*"  --rank=16
