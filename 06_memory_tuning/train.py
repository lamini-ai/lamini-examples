from typing import Generator, Any, Dict
import argparse
import jsonlines

from lamini import Lamini


def main() -> None:
    """Main runtime function for Instruction Fine Tuning

    Parameters
    ----------
    None

    Returns
    -------
    None
    """

    parser = argparse.ArgumentParser(description="Lamini training script.")
    parser.add_argument('--dataset-path', type=str,
                        default="data/results/generated_q_a.jsonl",
                        help='Path to the training dataset')
    args = parser.parse_args()

    llm = Lamini(model_name="meta-llama/Meta-Llama-3.1-8B-Instruct")

    dataset = list(load_training_data(args.dataset_path)) * 10

    llm.train(
        data_or_dataset_id=dataset,
        finetune_args={
            "max_steps": 300,
            "early_stopping": False,
            "load_best_model_at_end": False,
        },
    )


def load_training_data(path: str) -> Generator[Dict[str, Any], None, None]:
    """ Handler for jsonlines file data loading

    Parameters
    ----------
    path: str
        jsonline file location

    Yields
    -------
    Dict[str, Any]
        Dictionary of the input and expected output for each
        line within the provided path.
    """

    limit = 10

    with jsonlines.open(path) as reader:
        for index, obj in enumerate(reader):
            if index >= limit:
                break

            header = "<|begin_of_text|><|start_header_id|>user<|end_header_id|>"
            header_end = "<|eot_id|><|start_header_id|>assistant<|end_header_id|>"

            yield {
                "input": header + make_question(obj) + header_end,
                "output": obj["answer"] + "<|eot_id|>",
            }


def make_question(obj: Dict[str, Any]) -> str:
    """ Format a question string from the provided
    dictionary

    Parameters
    ----------
    obj: Dict[str, Any]
        Provided prompt metadata

    Returns
    -------
    str:
        Formatted new string
    """

    question = (
        f"Consider the following company: {obj['ticker']} and quarter: {obj['q']}. "
    )
    question += obj["question"]
    return question

if __name__ == "__main__":
    main()
