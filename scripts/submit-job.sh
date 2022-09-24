#!/bin/bash -l

#SBATCH --partition=cdac-own
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=10G
#SBATCH --signal=SIGUSR1@90
#SBATCH --output=slurm_stdout/%j.out
#SBATCH --open-mode=append
#SBATCH --exclude=a[006-008]
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=yimingz0@uchicago.edu

set -Eeuo pipefail

# export TRANSFORMERS_OFFLINE=1

python src/main.py "$@"
echo "Job finished!"