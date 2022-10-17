#!/bin/bash -l

#SBATCH --partition=cdac-contrib
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=30G
#SBATCH --signal=SIGUSR1@90
#SBATCH --output=slurm_stdout/%j.out
#SBATCH --open-mode=append
#SBATCH --exclude=a[001-004,006-008],aa[001-003],c001
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=yimingz0@uchicago.edu

set -Eeuo pipefail

python src/supervised-generation.py "$@"
echo "Job finished!"