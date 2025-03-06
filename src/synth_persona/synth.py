# Copyright 2025 Susumu OTA
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import logging
import re
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass
from time import time

import datasets
import transformers
from datasets import load_dataset
from litellm import batch_completion
from tqdm import tqdm
from transformers import set_seed
from trl import ModelConfig, ScriptArguments, TrlParser
from vllm import LLM, SamplingParams

logger = logging.getLogger(__name__)


@dataclass
class PersonaScriptArguments(ScriptArguments):
    logging_level: str = "INFO"
    dataset_batch_size: int = 64
    dataset_start: int = 0
    dataset_end: int = -1
    shuffle: bool = False
    cache_dir: str = ".cache"
    output_jsonl: str = "output.jsonl"

    max_tokens: int = 4096
    seed: int = 42
    temperature: float = 0.6
    top_p: float = 0.95

    task: str = "math problem"
    difficulty: str = "short, easy and involve basic mathematical skills and knowledge"
    target: str = "Any average grade school student"
    problem_start_with: str = "問題"
    problem_additional_note: str = "4. 深い専門知識が必要な問題を避け、平均的な知識と常識の範囲内で解ける問題にしてください。\n5. 簡潔に日本語で回答してください。"
    solution_start_with: str = "解答"
    solution_additional_note: str = "4. 簡潔に日本語で回答してください。"


class LanguageModel(ABC):
    def __init__(self, model: str, max_tokens=512, seed=None, temperature=1.0, top_p=0.95):
        self.model = model
        self.max_tokens = max_tokens
        self.seed = seed
        self.temperature = temperature
        self.top_p = top_p
        logger.debug(
            f"model: {model}, max_tokens: {max_tokens}, seed: {seed}, temperature: {temperature}, top_p: {top_p}"
        )

    @abstractmethod
    def __call__(self, messages_batch: list[list[dict[str, str]]]) -> list[str]:
        pass


class LiteLLMModel(LanguageModel):
    def __init__(self, model: str, max_tokens=512, seed=None, temperature=1.0, top_p=0.95):
        super().__init__(model, max_tokens, seed, temperature, top_p)

    def __call__(self, messages_batch: list[list[dict[str, str]]]) -> list[str]:
        contents = [
            response.choices[0].message.content or ""
            for response in batch_completion(
                model=self.model,
                messages=messages_batch,
                max_tokens=self.max_tokens,
                seed=self.seed,
                temperature=self.temperature,
                top_p=self.top_p,
            )
        ]
        assert len(contents) == len(messages_batch)
        return contents


class VLLMModel(LanguageModel):
    def __init__(self, model: str, max_tokens=512, seed=None, temperature=1.0, top_p=0.95):
        super().__init__(model, max_tokens, seed, temperature, top_p)
        self.vllm = LLM(model, seed=seed, gpu_memory_utilization=1.0, max_model_len=32 * 1024)  # TODO: parameterize
        self.tokenizer = self.vllm.get_tokenizer()

    def __call__(self, messages_batch: list[list[dict[str, str]]]) -> list[str]:
        sampling_params = SamplingParams(
            max_tokens=self.max_tokens, seed=self.seed, temperature=self.temperature, top_p=self.top_p
        )
        prompts = [
            self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            for messages in messages_batch
        ]
        outputs = self.vllm.generate(prompts, sampling_params=sampling_params, use_tqdm=False)
        contents = [output.outputs[0].text for output in outputs]
        assert len(contents) == len(messages_batch)
        return contents


def get_problem_prompt(
    persona: str,
    task: str,
    difficulty: str,
    target: str,
    start_with: str,
    additional_note: str,
) -> list[dict[str, str]]:
    template = """Create {a_an} {task} related to the following persona:

{persona}

Note:

1. The {task} should be {difficulty}. {target} can solve it correctly.
2. You should make full use of the persona description to create the {task} to ensure that the {task} is unique and specific to the persona.
3. Your response should always start with "{start_with}:". Your response should not include a solution to the created {task}.
{additional_note}
"""
    a_an = "an" if task[0].lower() in "aeiou" else "a"
    content = template.format(
        persona=persona,
        task=task,
        difficulty=difficulty,
        target=target,
        start_with=start_with,
        additional_note=additional_note,
        a_an=a_an,
    )
    return [{"role": "user", "content": content}]


def get_solution_prompt(problem: str, target: str, start_with: str, additional_note: str) -> list[dict[str, str]]:
    template = """Create a solution to the following problem:

{problem}

Note:

1. The solution should be concise, clean and easy to understand. {target} can understand it correctly.
2. You should make full use of the problem description to create the solution to ensure that the solution is unique and specific to the problem.
3. Your response should always start with "{start_with}:". Your response should only include the solution to the problem.
{additional_note}
"""
    content = template.format(
        problem=problem,
        target=target,
        start_with=start_with,
        additional_note=additional_note,
    )
    return [{"role": "user", "content": content}]


def parse_response(text: str) -> dict[str, str]:
    m = re.match(r"^(<think>)?(.*)</think>(.*)$", text, flags=re.DOTALL | re.MULTILINE)
    return (
        {"think": m.group(2), "answer": m.group(3), "format_reward": 1.0}
        if m
        else {"think": "", "answer": text, "format_reward": 0.0}
    )


def parse_answer(text: str, start_with: str) -> dict[str, str]:
    m = re.match(rf"^(.*){start_with}[:：](.*)$", text, flags=re.DOTALL | re.MULTILINE)
    answer = m.group(2).strip() if m else text
    answer = re.sub(r"^「(.*)」$", r"\1", answer).strip()
    return {"answer": answer, "extract_reward": 1.0 if m else 0.0}


def extract_answer(text: str, start_with: str) -> dict[str, str]:
    response = parse_response(text)
    answer = parse_answer(response["answer"], start_with)
    return {
        "think": response["think"],
        "answer": answer["answer"],
        "format_reward": response["format_reward"],
        "extract_reward": answer["extract_reward"],
    }


def generate_problems_solutions(
    llm: LanguageModel,
    indices: list[int],
    ids: list[str],
    personas: list[str],
    labelss: list[str],
    seed: int,
    task: str,
    difficulty: str,
    target: str,
    problem_start_with: str,
    problem_additional_note: str,
    solution_start_with: str,
    solution_additional_note: str,
) -> list[dict[str, str]]:
    # generate problems
    problem_prompts = [
        get_problem_prompt(persona, task, difficulty, target, problem_start_with, problem_additional_note)
        for persona in personas
    ]
    problems = [extract_answer(response, problem_start_with) for response in llm(problem_prompts)]

    # generate solutions
    solution_prompts = [
        get_solution_prompt(problem["answer"], target, solution_start_with, solution_additional_note)
        for problem in problems
    ]
    solutions = [extract_answer(response, solution_start_with) for response in llm(solution_prompts)]

    return [
        {
            "seed": seed,
            "index": index,
            "id": id,
            "persona": persona,
            "labels": labels,
            "problem_think": problem["think"],
            "problem_answer": problem["answer"],
            "problem_format_reward": problem["format_reward"],
            "problem_extract_reward": problem["extract_reward"],
            "solution_think": solution["think"],
            "solution_answer": solution["answer"],
            "solution_format_reward": solution["format_reward"],
            "solution_extract_reward": solution["extract_reward"],
        }
        for index, id, persona, labels, problem, solution in zip(
            indices,
            ids,
            personas,
            labelss,
            problems,
            solutions,
        )
    ]


def run_generate_problems_solutions(
    llm: LanguageModel,
    dataset: datasets.Dataset,
    seed: int,
    task: str,
    difficulty: str,
    target: str,
    problem_start_with: str,
    problem_additional_note: str,
    solution_start_with: str,
    solution_additional_note: str,
    dataset_batch_size: int = 8,
    dataset_start: int = 0,
    dataset_end: int = -1,
    output_jsonl: str = "output.jsonl",
):
    # ensure 0 <= start <= end <= len(dataset)
    end = len(dataset) if dataset_end == -1 else max(0, min(dataset_end, len(dataset)))
    start = min(max(0, dataset_start), end)
    with open(output_jsonl, "w", encoding="utf-8") as f:
        for batch_start in tqdm(range(start, end, dataset_batch_size)):
            time_start = time()
            batch_end = min(batch_start + dataset_batch_size, end)
            batch = dataset[batch_start:batch_end]
            if len(batch) == 0:
                break
            indices = list(range(batch_start, batch_end))
            ids = batch["id"]
            personas = batch["persona"]
            labelss = batch["labels"]
            logger.debug(
                f"load data: seed: {seed}: batch: {batch_start}-{batch_end}: ({(time() - time_start):.4f} sec)"
            )
            time_start = time()
            problems = generate_problems_solutions(
                llm,
                indices,
                ids,
                personas,
                labelss,
                seed,
                task,
                difficulty,
                target,
                problem_start_with,
                problem_additional_note,
                solution_start_with,
                solution_additional_note,
            )
            logger.debug(f"generate_problems: ({(time() - time_start):.4f} sec)")
            time_start = time()
            for problem in problems:
                f.write(json.dumps(problem, ensure_ascii=False) + "\n")
                f.flush()
            logger.debug(f"json dumps: {batch_start}-{batch_end} ({(time() - time_start):.4f} sec)")


def main(script_args: PersonaScriptArguments, model_args: ModelConfig):
    # Set seed
    set_seed(script_args.seed)

    # Set logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger.setLevel(script_args.logging_level)
    datasets.utils.logging.set_verbosity(script_args.logging_level)
    transformers.utils.logging.set_verbosity(script_args.logging_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()
    logger.debug(f"script_args: {script_args}")
    logger.debug(f"model_args: {model_args}")

    # Load dataset
    dataset = load_dataset(
        script_args.dataset_name,
        name=script_args.dataset_config,
        split=script_args.dataset_train_split,
        cache_dir=script_args.cache_dir,
    )
    logger.debug(f"Dataset: {dataset}")

    if script_args.shuffle:
        logger.debug(f"Shuffling dataset with seed {script_args.seed}...")
        dataset = dataset.shuffle(seed=script_args.seed)
        logger.debug(f"Shuffling dataset with seed {script_args.seed}...done")

    # Load language model
    llm = VLLMModel(
        model_args.model_name_or_path,
        seed=script_args.seed,
        max_tokens=script_args.max_tokens,
        temperature=script_args.temperature,
        top_p=script_args.top_p,
    )
    logger.debug(f"LLM: {llm}")

    # Generate problems
    run_generate_problems_solutions(
        llm,
        dataset,
        script_args.seed,
        script_args.task,
        script_args.difficulty,
        script_args.target,
        script_args.problem_start_with,
        script_args.problem_additional_note,
        script_args.solution_start_with,
        script_args.solution_additional_note,
        dataset_batch_size=script_args.dataset_batch_size,
        dataset_start=script_args.dataset_start,
        dataset_end=script_args.dataset_end,
        output_jsonl=script_args.output_jsonl,
    )


if __name__ == "__main__":
    parser = TrlParser((PersonaScriptArguments, ModelConfig))
    script_args, model_args = parser.parse_args_and_config()
    main(script_args, model_args)
