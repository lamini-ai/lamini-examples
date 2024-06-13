from lamini import Lamini

import jsonlines


def main():
    llm = Lamini(model_name="meta-llama/Meta-Llama-3-8B-Instruct")

    dataset = list(load_training_data()) * 10

    llm.train(
        data_or_dataset_id=dataset,
        finetune_args={
            "max_steps": 300,
            "early_stopping": False,
            "load_best_model_at_end": False,
        },
    )


def load_training_data():
    path = "/app/lamini-earnings-sdk/data/results/generated_q_a.jsonl"

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


def make_question(obj):
    question = (
        f"Consider the following company: {obj['ticker']} and quarter: {obj['q']}. "
    )
    question += obj["question"]
    return question


main()
