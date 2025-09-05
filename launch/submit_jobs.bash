#!/bin/bash
#SBATCH --requeue
#SBATCH --output=slurm/%j.out
#SBATCH --error=slurm/%j.err
#SBATCH --open-mode=append
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-gpu=8
#SBATCH --time=2880
#SBATCH --partition=3po,devaccel,scavenge
#SBATCH --signal=SIGUSR1@90
#SBATCH --constraint=volta32gb

# srun python $@ hydra.job.id=$(date '+%Y%m%d%H%M%S')-$SLURM_JOB_ID +job_id=$SLURM_JOB_ID +branch=$(git rev-parse --abbrev-ref HEAD) +commit=$(git rev-parse --short HEAD)
srun python $@ 