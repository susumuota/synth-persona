# Synthetic text using Persona

## Setup

```shell
module load cuda/12.4

uv sync
```

## Run

- With Slurm

```shell
sbatch slurm/synth.slurm
```

- Withouth Slurm

```shell
CUDA_VISIBLE_DEVICES=0 uv run src/synth_persona/synth.py --config receipes/Qwen2.5-0.5B-Instruct/config_finepersonas.yaml

# TODO
uv run accelerate launch --config_file receipes/accelerate_configs/zero2.yaml src/synth_persona/synth.py --config receipes/Qwen2.5-0.5B-Instruct/config_finepersonas.yaml
uv run accelerate launch --config_file receipes/accelerate_configs/cpu.yaml src/synth_persona/synth.py --config receipes/Qwen2.5-0.5B-Instruct/config_finepersonas.yaml
```
