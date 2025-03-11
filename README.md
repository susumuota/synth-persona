# Synthetic Data Creation with Personas

## Setup

```shell
module load cuda/12.4  # this depends on your environment

git clone https://github.com/susumuota/synth-persona.git
cd synth-persona

# install uv if you haven't
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync
```

## Run

- With Slurm

```shell
sbatch slurm/synth.slurm
sbatch slurm/synth_api.slurm

sbatch slurm/extract.slurm
sbatch slurm/extract_api.slurm
```

- Without Slurm

```shell
CUDA_VISIBLE_DEVICES=0 uv run src/synth_persona/synth.py --config recipes/deepseek-r1-distill-qwen2.5-bakeneko-32b/config_finepersonas.yaml --output-jsonl outputs/output-1.jsonl --seed 1 --shuffle true
```

```shell
uv run src/synth_persona/extract.py --config recipes/gpt-4o-mini/config_fineweb_2.yaml --output-jsonl outputs/output-1.jsonl --seed 1 --shuffle true
```

## Check logs

```shell
tail -f logs/synth-persona-{jobid}.out
tail -f logs/synth-persona-{jobid}.err
```

## Confirm the output

- Check the uniqueness of the output

```shell
# $6 should be the UUID
cat outputs/output-*.jsonl | awk '{ print $6 }' | sort | uniq -c | sort -n
```

- Check the content of the output

```shell
# find the record with the UUID
grep -h "ce349faf-3803-44de-a8c4-cbed02365ab1" outputs/*.jsonl | jq -C | less -R
```

- Check the reward of the output

```shell
# find the record with the reward 0.0
grep --color=always "reward\": 0\.0" outputs/output-*.jsonl | less -R
```
