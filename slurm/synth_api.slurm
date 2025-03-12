#!/bin/bash
#SBATCH --job-name=synth-persona
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:0
#SBATCH --partition=hopper-prod  # Adjust this for your cluster
#SBATCH --output=./logs/%x-%j.out
#SBATCH --err=./logs/%x-%j.err

echo "START TIME: $(date)"

# module load cuda/12.4

uv run src/synth_persona/synth.py --config recipes/gpt-4o-mini/config_finepersonas.yaml

echo "END TIME: $(date)"
