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

from argparse import ArgumentParser
from logging import DEBUG, StreamHandler, getLogger

from datasets import load_dataset
from vllm import LLM, SamplingParams

logging_level = DEBUG
# logging_level = INFO  # uncomment if you want to see less output
logger = getLogger(__name__)
logger.setLevel(logging_level)
handler = StreamHandler()
handler.setLevel(logging_level)
logger.addHandler(handler)


def main():
    parser = ArgumentParser()
    parser.add_argument("--dataset", type=str, default="argilla/FinePersonas-v0.1")
    parser.add_argument("--model", type=str, default="rinna/deepseek-r1-distill-qwen2.5-bakeneko-32b")
    args = parser.parse_args()
    dataset = load_dataset(args.dataset, "default", split="train", cache_dir="cache")
    logger.debug(f"dataset: {dataset}")

    llm = LLM(args.model, gpu_memory_utilization=1.0, max_model_len=1024 * 32)
    logger.debug(f"llm: {llm}")
    sampling_params = SamplingParams(temperature=0.6, max_tokens=512, seed=0)
    prompt = llm.get_tokenizer().apply_chat_template(
        [{"role": "user", "content": "数学の問題を作ってください。"}], tokenize=False, add_generation_prompt=True
    )
    logger.debug(f"prompt: {prompt}")
    outputs = llm.generate([prompt], sampling_params=sampling_params, use_tqdm=False)
    logger.debug(f"len(outputs): {len(outputs)}")
    contents = [o.outputs[0].text for o in outputs]
    logger.debug(f"contents: {contents}")


if __name__ == "__main__":
    main()
