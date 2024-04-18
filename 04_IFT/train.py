import jsonlines
from lamini import MistralRunner


def main():
    runner = MistralRunner(
        system_prompt=" ",
    )

    data = list(load_data())

    runner.load_data(
        data=data,
        input_key="question",
        output_key="answer",
    )

    runner.train()


def load_data():
    path = "/app/lamini-ift/qa_data/generated_data_finetuning.jsonl"

    with jsonlines.open(path) as reader:
        for obj in reader:
            yield {"question": obj["question"], "answer": obj["answer"] + "</s>"}


main()
