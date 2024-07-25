from typing import AsyncGenerator, List

import jsonlines
import os
import logging

from argparse import ArgumentParser, Namespace

from load_earnings_call_dataset import load_earnings_call_dataset, EarningsCallsDataset
from lamini.generation.base_prompt_object import PromptObject
from eval_pipeline import evaluate_model


def main() -> None:
    """ Main script for eval runtime

    Parameters
    ----------
    None

    Returns
    -------
    None
    """
    args = parse_arguments()

    setup_logging(args)

    dataset = slice_dataset(load_dataset(args), args.max_examples)

    results = evaluate_model(dataset, args)

    save_results(results, args)


async def slice_dataset(
        dataset: EarningsCallsDataset, 
        max_examples: int
    ) -> AsyncGenerator[PromptObject, None, None]:
    """ Enforce the max_examples limit on the provided
    dataset

    Parameters
    ----------
    dataset: EarningsCallsDataset
        Dataset loader

    max_examples: int
        Upper limit of example count
        
    Yields
    ------
    PromptObject
        Construct prompt from loaded example in dataset

    """

    for index, example in enumerate(dataset):
        if index < max_examples:
            yield PromptObject(prompt=example.get_prompt(), data={"example": example})


def parse_arguments() -> Namespace:
    """ Argument Parser setup 
    The following arguments are used in this test script:
        --data
            Path and file name for the evaluation data

        --model
            Name of the model to evaluate
            default meta-llama/Meta-Llama-3-8B-Instruct

        --max-examples
            Max number of examples to evaluate
            default 100
        
    Returns
    -------
    argparse.Namespace
        Namespace object storing the given args as attributes
    """

    parser = ArgumentParser()

    parser.add_argument(
        "--data",
        type=str,
        default="earnings",
        help="The name of the evaluation dataset",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="meta-llama/Meta-Llama-3-8B-Instruct",
        help="The name of the model to evaluate",
    )

    # The max number of examples to evaluate
    parser.add_argument(
        "--max-examples",
        type=int,
        default=100,
        help="The max number of examples to evaluate",
    )

    return parser.parse_args()


def setup_logging(args: Namespace) -> None:
    """ Establish logging configuration and starting log

    Parameters
    ----------
    args: Namespace
        Input arguments to the main script
        The following values are used:
            data
                Path and file name for the evaluation data

            model
                Name of the model to evaluate
                default meta-llama/Meta-Llama-3-8B-Instruct

    Returns
    -------
    None
    """

    # Remove all handlers associated with the root logger object.
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler()],
    )

    logging.info(f"Evaluating model {args.model} on dataset {args.data}")


def load_dataset(args: Namespace) -> EarningsCallsDataset:
    """ Logic for which dataset to return. Specific loading
    functions exist for different data key words in args.

    Parameters
    ----------
    args: Namespace
        Input arguments to the main script
        The following values are used:
            data
                Path and file name for the evaluation data

    Returns
    -------
    EarningsCallsDataset
        Dataset returned from the specified loading function

    Raises
    ------
    ValueError
        Raised if the dataset is not found in the pre-set 
        dataset key words
    """

    if args.data == "earnings":
        return load_earnings_call_dataset()
    else:
        raise ValueError(f"Unknown dataset: {args.data}")


def save_results(results: List[PromptObject], args: Namespace) -> None:
    """ Store results in provided path in args

    Parameters
    ----------
    results: List[PromptObject]
        Return list of query results to LLM

    args: Namespace
        Input arguments to the main script
        The following values are used:
            data
                Path and file name for the evaluation data
            model
                Name of the model to evaluate
                default meta-llama/Meta-Llama-3-8B-Instruct

    Returns
    -------
    None
    """

    base_path = "/app/lamini-earnings-sdk/data/results"
    experiment_name = f"{args.data}_{args.model}".replace("/", "_")

    if not os.path.exists(base_path):
        os.makedirs(base_path)

    file_name = f"{base_path}/{experiment_name}_results.json"

    with jsonlines.open(file_name, "w") as writer:
        for result in results:
            writer.write(
                {
                    "id": result.data["result"]["example_id"],
                    "prompt": result.data["result"]["prompt"],
                    "response": result.data["result"]["response"],
                    "reference_response": result.data["result"]["reference_response"],
                    "is_exact_match": result.data["result"]["is_exact_match"],
                    "score": result.data["result"]["score"],
                    "explanation": result.data["result"]["explanation"],
                }
            )

if __name__ == "__main__":
    main()
