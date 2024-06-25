from typing import List

from lamini_rag.lamini_index import LaminiIndex
from lamini_rag.earnings_call_loader import EarningsCallLoader
from lamini_rag.example_prompt_formater import EarningsExample
from lamini_rag.lamini_pipeline import SpotCheckPipeline, DatasetDescriptor

from lamini.generation.base_prompt_object import PromptObject

from tqdm import tqdm

from argparse import ArgumentParser, Namespace

import asyncio
import jsonlines

import os

import logging

logger = logging.getLogger(__name__)

def setup_logging() -> None:
    """ Logging setup """

    # Remove all handlers associated with the root logger object.
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler()],
    )


def parse_arguments() -> Namespace:
    """ Argument Parser setup 
    The following arguments are used in this test script:
        --max-examples
            Max number of examples to load within load_dataset.
            These examples are used for testing evaluation of 
            the SpotCheckPipeline after the index has been loaded
            or built

        --rag-path
            This is the path to the rag index if it already exists.
            If the index does not exist, then this path is used as
            the location for which to store the newly built index. 
            Existing indices are expected to be in the faiss format

        --index-data
            This is the path to the data to use for building a rag
            index (if an existing one was not provided). All paths
            for this example are expected to be in the jsonl format

        --test-data
            This is the path to the data used for evaluating the 
            generated responses from the model. Both question and answer
            are expected within this data. All paths for this argument 
            are expected to be in the jsonl format

        --output-path
            This is the path that will be storing the output results
            when calling the pipeline with the supplied data from
            --test-data. Data will be stored in the jsonl format
        
    Returns
    -------
    argparse.Namespace
        Namespace object storing the given args as attributes
    """

    parser = ArgumentParser()

    # The max number of examples to evaluate
    parser.add_argument(
        "--max-examples",
        type=int,
        default=3,
        help="The max number of examples to evaluate",
    )

    # The path to load and save RAG Indices
    parser.add_argument(
        "--rag-path",
        type=str,
        default="/app/lamini-earnings-sdk/models/",
        help="Path of existing RAG Indices or where new ones are saved",
    )

    # The path to test data used to build index
    parser.add_argument(
        "--index-data",
        type=str,
        default="/app/lamini-earnings-sdk/data/test_set_transcripts.jsonl",
        help="Path of test data used to build RAG index",
    )

    # The path to test data used to build index
    parser.add_argument(
        "--test-data",
        type=str,
        default="/app/lamini-earnings-sdk/data/golden_test_set.jsonl",
        help="Path of test data used to evaluate the model generated responses",
    )

    # The output path to store the generated results
    parser.add_argument(
        "--output-path",
        type=str,
        default="/app/lamini-earnings-sdk/data/results/rag_spot_check_results.jsonl",
        help="Path for results of model generation to be stored",
    )

    return parser.parse_args()


async def build_rag_index(args: Namespace) -> None:
    """ Build/Load Rag Index
    Load in a rag space from the provided args.rag_path argument. If
    the provided path in args.rag_path does not exist, then build the index
    from the provided data located in the path args.index_data.

    Parameters
    ----------
    args: argparse.Namespace
        Namespace object storing
            rag_path: path string to the rag index storage loaction
            index_data: path string to the data used to build rag index

    """

    if os.path.exists(os.path.join(args.rag_path, "index.faiss")):
        logger.info("Index already exists, loading it...")
        return

    logger.info("Building index")
    # Build the index if it doesn't exist
    index = LaminiIndex(loader=EarningsCallLoader(path=args.index_data))
    await index.build_index()

    os.makedirs(args.rag_path, exist_ok=True)
    index.save_index(args.rag_path)


def load_dataset(args: Namespace) -> PromptObject:
    """Load in test dataset
    Load in the test data from the args.test_data attribute, expected
    format is to be in jsonl.

    Each data point in the test_data is then added into a EarningExample 
    object which is then wrapped into a PromptObject to be yielded back
    to the returning function

    Yield is used as this function is expected to be used within an async 
    call to the Lamini platform.

    Parameters
    ----------
    args: argparse.Namespace
        Namespace object storing
            test_data: path string to the evaluation data
            max_examples: limit of number of examples to yield

    Yields
    ------
    lamini.generation.base_prompt_object.PromptObject
        PromptObject storing the single EarningsExamples prompt and data
    """
    
    with jsonlines.open(args.test_data) as reader:
        dataset = list(reader)

    for index, example in enumerate(dataset):
        if index < args.max_examples:
            earnings_example = EarningsExample(example)
            yield PromptObject(
                prompt=earnings_example.get_prompt(), data=earnings_example
            )


async def run_spot_check(args) -> List[PromptObject]:
    """ Main runtime function to run the spot check for RAG 

    Parameters
    ----------
    args: argparse.Namespace
        Namespace object storing user provided arguments

    Returns
    -------
    List[lamini.generation.base_prompt_object.PromptObject]
        PromptObjects returned from the SpotCheckPipeline are wrapped
        into a list
    """

    dataset = load_dataset(args)

    results = SpotCheckPipeline(DatasetDescriptor()).call(dataset)

    result_list = []

    pbar = tqdm(desc="Saving results", unit=" results")
    async for result in results:
        result_list.append(result)
        pbar.update()

    return result_list


def save_results(args, results) -> None:
    """Store the generated results into the provided output path

    Parameters
    ----------
    args: argparse.Namespace
        Namespace object storing
            output_path: path string for the output file

    results: List[lamini.generation.base_prompt_object.PromptObject]
        Results from the generation pipeline SpotCheckPipeline
    """

    with jsonlines.open(args.output_path, "w") as writer:
        for result in results:

            row = result.data.example.copy()
            row["model_answer"] = result.response["model_answer"]
            row["prompt"] = result.prompt

            writer.write(row)

            print("\n\n")
            print(result.prompt, "\n")
            print("Model Answer: ", result.response["model_answer"])
            print("\n")


if __name__ == "__main__":
    # Run preprocess functions of initalizing logging and argument parsing 
    setup_logging()

    args = parse_arguments()
    # Run function to find and load index or build a new index
    asyncio.run(build_rag_index(args))

    # Run the spot check pipeline with the loaded (or newly built) index
    results = asyncio.run(run_spot_check(args))

    # Store results in --output-path
    save_results(args, results)
