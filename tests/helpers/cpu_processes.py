import os
import sys

# start 2 processes on the CPU with torchrun. GPUs are hidden: Accelerate >= 1.x passes the local
# rank as a CUDA device to `barrier()` even for CPU (gloo) runs, which fails when GPUs are visible.
TORCHRUN_2_CPU = [
    sys.executable,
    "-m",
    "torch.distributed.run",
    "--standalone",
    "--nproc_per_node=2",
]
CPU_ONLY_ENV = {**os.environ, "CUDA_VISIBLE_DEVICES": ""}
