#!/bin/bash
#SBATCH --job-name=synth-persona
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --partition=hopper-prod  # Adjust this for your cluster
#SBATCH --output=./logs/%x-%j.out
#SBATCH --err=./logs/%x-%j.err

echo "START TIME: $(date)"

module load cuda/12.4

uv run src/synth_persona/synth.py --config receipes/deepseek-r1-distill-qwen2.5-bakeneko-32b/config_finepersonas.yaml --output-jsonl outputs/output-${SLURM_JOB_ID}.jsonl --seed ${SLURM_JOB_ID} --shuffle true

echo "END TIME: $(date)"
