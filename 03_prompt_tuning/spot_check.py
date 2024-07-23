from typing import Union, Iterator, AsyncIterator


from lamini.generation.base_prompt_object import PromptObject
from lamini.generation.generation_pipeline import GenerationPipeline
from lamini.generation.generation_node import GenerationNode


from tqdm import tqdm

from argparse import ArgumentParser

import asyncio
import jsonlines

import logging


def main():
    setup_logging()

    args = parse_arguments()

    results = asyncio.run(run_spot_check(args))

    save_results(results)


def setup_logging():
    # Remove all handlers associated with the root logger object.
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler()],
    )


def parse_arguments():
    parser = ArgumentParser()

    # The max number of examples to evaluate
    parser.add_argument(
        "--max-examples",
        type=int,
        default=3,
        help="The max number of examples to evaluate",
    )

    return parser.parse_args()


async def run_spot_check(args):
    dataset = load_dataset(args)

    results = SpotCheckPipeline().call(dataset)

    result_list = []

    pbar = tqdm(desc="Saving results", unit=" results")
    async for result in results:
        result_list.append(result)
        pbar.update()

    return result_list


class DatasetDescriptor:
    def get_output_type(self):
        return {"model_answer": "str"}


def load_dataset(args):
    path = "/Users/yaxiong/Workspace/lamini-ai/lamini-examples/data/golden_test_set.jsonl"

    with jsonlines.open(path) as reader:
        dataset = list(reader)

    for index, data in enumerate(dataset):
        if index < args.max_examples:
            yield PromptObject(
                prompt="", data=data
            )


def make_prompt(example):
    prompt = "You are an expert analyst from Goldman Sachs with 15 years of experience."
    prompt += " Consider the following company: \n"
    prompt += "==========================\n"
    prompt += get_company_info(example)
    prompt += "==========================\n"
    prompt += "Answer the following question: \n"
    prompt += example["question"]
    return prompt


def get_company_info(example):
    prompt = f"Date of the call: {example['date']}\n"
    prompt += f"Ticker: {example['ticker']}\n"
    prompt += f"Quarter: {example['q']}\n"

    return prompt


class LaminiModelStage(GenerationNode):
    def __init__(self, model_name="meta-llama/Meta-Llama-3-8B-Instruct",):
        super().__init__(
            model_name=model_name,
            max_new_tokens=150,
        )
        self.model_name = model_name

    def preprocess(self, prompt: PromptObject):
        new_prompt = "<|begin_of_text|><|start_header_id|>user<|end_header_id|>"
        new_prompt += make_prompt(prompt.data) + "<|eot_id|>"
        new_prompt += "<|start_header_id|>assistant<|end_header_id|>"
        prompt.prompt = new_prompt


class SpotCheckPipeline(GenerationPipeline):
    def __init__(self):
        super().__init__()
        self.model_stage = LaminiModelStage()

    def forward(self, x):
        x = self.model_stage(x, output_type={"model_answer": "str"})
        return x


def save_results(results):
    file_name = "spot_check_results.jsonl"

    with jsonlines.open(file_name, "w") as writer:
        for result in results:

            row = result.data.copy()
            row["model_answer"] = result.response["model_answer"]
            row["prompt"] = result.prompt

            writer.write(row)

            print("\n\n")
            print(result.prompt, "\n")
            print("Model Answer: ", result.response["model_answer"])
            print("\n")


main()
