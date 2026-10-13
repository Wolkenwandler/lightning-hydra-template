#!/bin/sh
# Start a multi-process run (e.g. several GPUs) with `accelerate launch`.
#
# Every process composes the Hydra config itself; RUN_TIMESTAMP gives all of them the same
# run directory (see configs/hydra/default.yaml).
#
# usage: scripts/launch.sh [accelerate launch options] src/train.py [hydra overrides]
#   e.g. scripts/launch.sh --multi_gpu --num_processes 8 src/train.py experiment=example
RUN_TIMESTAMP=${RUN_TIMESTAMP:-$(date +%Y-%m-%d_%H-%M-%S)}
export RUN_TIMESTAMP
exec accelerate launch "$@"
