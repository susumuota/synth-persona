#!/bin/bash
#SBATCH --job-name=synth-persona
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --partition=hopper-prod  # Adjust this for your cluster
#SBATCH --output=./logs/%x-%j.out
#SBATCH --err=./logs/%x-%j.err

echo "START TIME: $(date)"

module load cuda/12.4

# uv run src/synth_persona/extract.py --config recipes/deepseek-r1-distill-qwen2.5-bakeneko-32b/config_fineweb_2.yaml
uv run src/synth_persona/extract.py --config recipes/deepseek-r1-distill-qwen2.5-bakeneko-32b/config_fineweb_2_en.yaml

echo "END TIME: $(date)"
