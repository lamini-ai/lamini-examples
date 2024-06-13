from lamini_rag.lamini_embedding_model_stage import LaminiEmbeddingModelStage
from lamini_rag.lamini_rag_model_stage import LaminiRAGModelStage

from lamini_rag.lamini_index import LaminiIndex
from lamini_rag.earnings_call_loader import EarningsCallLoader

from lamini.generation.base_prompt_object import PromptObject
from lamini.generation.generation_pipeline import GenerationPipeline

from tqdm import tqdm

from argparse import ArgumentParser

import asyncio
import jsonlines

import os

import logging

logger = logging.getLogger(__name__)

def main():
    setup_logging()

    args = parse_arguments()

    asyncio.run(build_rag_index(args))

    results = asyncio.run(run_spot_check(args))

    save_results(results)


def setup_logging():
    # Remove all handlers associated with the root logger object.
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    logging.basicConfig(
        level=logging.WARNING,
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

async def build_rag_index(args):
    model_path = "/app/lamini-earnings-sdk/models/"

    if os.path.exists(os.path.join(model_path, "index.faiss")):
        logger.info("Index already exists, loading it...")
        return

    logger.info("Building index")
    # Build the index if it doesn't exist
    path = "/app/lamini-earnings-sdk/data/test_set_transcripts.jsonl"

    index = LaminiIndex(loader=EarningsCallLoader(path=path))
    await index.build_index()

    os.makedirs(model_path, exist_ok=True)
    index.save_index(model_path)

async def run_spot_check(args):
    dataset = load_dataset(args)

    results = SpotCheckPipeline(DatasetDescriptor()).call(dataset)

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
    path = "/app/lamini-earnings-sdk/data/golden_test_set.jsonl"

    with jsonlines.open(path) as reader:
        dataset = list(reader)

    for index, example in enumerate(dataset):
        if index < args.max_examples:
            earnings_example = EarningsExample(example)
            yield PromptObject(
                prompt=earnings_example.get_prompt(), data=earnings_example
            )


class EarningsExample:
    def __init__(self, example):
        self.example = example

    def get_prompt(self):
        return make_prompt(self.example)

    def get_query(self):
        prompt = get_company_info(self.example)
        prompt += self.example["question"]

        return prompt


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


class SpotCheckPipeline(GenerationPipeline):
    def __init__(self, dataset):
        super().__init__()
        self.embedding_stage = LaminiEmbeddingModelStage(dataset)
        self.model_stage = LaminiRAGModelStage(dataset)

    def forward(self, x):
        x = self.embedding_stage(x)
        x = self.model_stage(x)
        return x


def save_results(results):
    file_name = "/app/lamini-earnings-sdk/data/results/rag_spot_check_results.jsonl"

    with jsonlines.open(file_name, "w") as writer:
        for result in results:

            row = result.data.example.copy()
            row["model_answer"] = result.response["model_answer"]
            row["prompt"] = result.prompt

            writer.write(row)

            print("\n\n")
            print(result.prompt, "\n")
            print("Model Answer: ", result.response["model_answer"])
            print("\n")


main()
