from lamini import MistralRunner
import jsonlines


def main():
    llm = MistralRunner(
        system_prompt=" ",
    )

    data = list(load_data())

    llm.load_data(
        data=data,
        input_key="question",
        output_key="answer",
    )

    llm.train()


def load_data():
    path = "/app/lamini-ift/data/questions_and_answers.jsonl"

    with jsonlines.open(path) as reader:
        for obj in reader:
            yield {"question": obj["question"], "answer": obj["answer"] + "</s>"}


main()
