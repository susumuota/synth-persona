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

import logging
import sys
from dataclasses import dataclass

import datasets
import transformers
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import ModelConfig, ScriptArguments, TrlParser

logger = logging.getLogger(__name__)


@dataclass
class SynthPersonaArguments(ScriptArguments):
    logging_level: str = "INFO"
    cache_dir: str = "cache"
    topic: str = "math"


def main(script_args: SynthPersonaArguments, model_args: ModelConfig):
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

    dataset = load_dataset(
        script_args.dataset_name,
        name=script_args.dataset_config,
        split=script_args.dataset_train_split,
        cache_dir=script_args.cache_dir,
    )
    logger.debug(f"Dataset: {dataset}")
    logger.debug(f"dataset[0]: {dataset[0]}")

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=model_args.trust_remote_code,
        use_fast=True,
        cache_dir=script_args.cache_dir,
    )
    logger.debug(f"Tokenizer: {tokenizer}")

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        revision=model_args.model_revision,
        torch_dtype=model_args.torch_dtype,
        trust_remote_code=model_args.trust_remote_code,
        attn_implementation=model_args.attn_implementation,
        cache_dir=script_args.cache_dir,
    )
    logger.debug(f"Model: {model}")

    messages_batch = [
        [
            {"role": "user", "content": "Hello!"},
        ],
        [
            {"role": "user", "content": "Hey!"},
        ],
    ]
    prompts = [
        tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        for messages in messages_batch
    ]
    logger.debug(f"Prompts: {prompts}")

    outputs = model.generate(
        **tokenizer(prompts, return_tensors="pt"),
        max_length=100,
        num_return_sequences=1,
        do_sample=True,
        temperature=0.7,
        top_k=50,
        top_p=0.95,
    )
    logger.debug(f"Outputs: {outputs}")

    responses = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    logger.debug(f"Responses: {responses}")


if __name__ == "__main__":
    parser = TrlParser((SynthPersonaArguments, ModelConfig))
    script_args, model_args = parser.parse_args_and_config()
    main(script_args, model_args)
