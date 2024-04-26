import jsonlines
from lamini import Lamini

def main():
    llm = Lamini(model_name="mistralai/Mistral-7B-Instruct-v0.2")
    data = list(load_data())
    result = llm.train(data)

def load_data():
    path = "qa_data/generated_data_finetuning.jsonl"
    with jsonlines.open(path) as reader:
        for obj in reader:
            yield {"input": obj["question"], "output": obj["answer"] + "</s>"}
main()
