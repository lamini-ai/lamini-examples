
import jsonlines
from tqdm import tqdm

def main():
    examples = load_examples()

    save_filtered_examples(examples)

def load_examples():
    path = "/app/lamini-classify/data/predictions.jsonl"

    with jsonlines.open(path) as reader:
        for example in reader:
            yield example

def save_filtered_examples(examples):
    threshold = 0.49831711334318995

    positive_examples = []
    for example in tqdm(examples):
        predictions = example["predictions"]

        positive_class = get_positive_class(predictions)

        if positive_class["prob"] >= threshold:
            positive_examples.append(example)

    # Sort the examples by the probability of the positive class
    positive_examples.sort(key=lambda example: get_positive_class(example["predictions"])["prob"], reverse=True)

    with jsonlines.open("/app/lamini-classify/data/filtered_predictions.jsonl", mode="w") as writer:
        for example in tqdm(positive_examples):
            writer.write(example)

    print(f"Saved {len(positive_examples)} examples")


def get_positive_class(predictions):
    for prediction in predictions:
        if prediction["class_name"] == "correct":
            return prediction

main()

