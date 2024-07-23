import jsonlines
import os
import logging

from argparse import ArgumentParser
from typing import Union, Iterator, AsyncIterator

from load_earnings_call_dataset import load_earnings_call_dataset, EarningsCallsExample
from lamini.generation.generation_node import GenerationNode
from lamini.generation.base_prompt_object import PromptObject
from eval_pipeline import evaluate_model


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


class LaminiModelStage(GenerationNode):
    def __init__(self, dataset, model_name="meta-llama/Meta-Llama-3-8B-Instruct",):
        super().__init__(
            model_name=model_name,
            max_new_tokens=150,
        )
        self.model_name = model_name
        self.dataset = dataset

    def generate(
        self,
        prompt: Union[Iterator[PromptObject], AsyncIterator[PromptObject]],
        *args,
        **kwargs,
    ):
        results = super().generate(
            prompt,
            output_type=self.dataset.get_output_type(),
            *args,
            **kwargs,
        )

        return results

    def preprocess(self, prompt: PromptObject):
        example = prompt.data["example"]
        new_prompt = "<|begin_of_text|><|start_header_id|>user<|end_header_id|>"
        new_prompt += example.get_prompt() + "<|eot_id|>"
        new_prompt += "<|start_header_id|>assistant<|end_header_id|>"

        return PromptObject(prompt=new_prompt, data=prompt.data)


def load_lamini_model(model_name):
    return LaminiModel(model_name)


class LaminiModel:
    def __init__(self, model_name):
        self.model_name = model_name
        if model_name is None:
            self.model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
    def get_stages(self, dataset):
        return [LaminiModelStage(dataset=dataset, model_name=self.model_name)]


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
                    "id": result.data["result"]["example_id"],
                    "prompt": result.data["result"]["prompt"],
                    "response": result.data["result"]["response"],
                    "reference_response": result.data["result"]["reference_response"],
                    "is_exact_match": result.data["result"]["is_exact_match"],
                    "score": result.data["result"]["score"],
                    "explanation": result.data["result"]["explanation"],
                }
            )


main()
