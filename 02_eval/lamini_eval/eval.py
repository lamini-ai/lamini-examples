from argparse import ArgumentParser

from lamini_eval.datasets.earnings_call_dataset import load_earnings_call_dataset
from lamini_eval.models.lamini_model import load_lamini_model

from lamini_eval.eval_pipeline import evaluate_model

import jsonlines

import os

import logging


def main():
    args = parse_arguments()

    setup_logging(args)

    dataset = load_dataset(args)

    model = load_model(args)

    results = evaluate_model(model, dataset, args)

    save_results(results, args)


def parse_arguments():
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


def setup_logging(args):
    # Remove all handlers associated with the root logger object.
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler()],
    )

    logging.info(f"Evaluating model {args.model} on dataset {args.data}")


def load_dataset(args):
    if args.data == "earnings":
        return load_earnings_call_dataset()
    else:
        raise ValueError(f"Unknown dataset: {args.data}")


def load_model(args):
    return load_lamini_model(args.model)


def save_results(results, args):
    base_path = "/app/lamini-earnings-sdk/data/results"
    experiment_name = f"{args.data}_{args.model}".replace("/", "_")

    if not os.path.exists(base_path):
        os.makedirs(base_path)

    file_name = f"{base_path}/{experiment_name}_results.json"

    with jsonlines.open(file_name, "w") as writer:
        for result in results:
            writer.write(
                {
                    "id": result.data.result["example_id"],
                    "prompt": result.data.result["prompt"],
                    "response": result.data.result["response"],
                    "reference_response": result.data.result["reference_response"],
                    "is_exact_match": result.data.result["is_exact_match"],
                    "score": result.data.result["score"],
                    "explanation": result.data.result["explanation"],
                }
            )


main()

