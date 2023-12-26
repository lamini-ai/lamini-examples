
from lamini_classifier import LaminiClassifier

from tqdm import tqdm
import jsonlines

batch_size = 128

def main():
    classifier = LaminiClassifier.load("models/classifier.lamini")

    examples = load_examples()

    predictions = predict(classifier, examples)

    save_predictions(predictions)

def load_examples():
    path = "/app/lamini-classify/data/questions_and_answers.jsonl"

    with jsonlines.open(path) as reader:
        for example in reader:
            yield example

def predict(classifier, examples):
    example_batches = batch(examples)

    for example_batch in example_batches:
        predictions = predict_batch(classifier, example_batch)

        for prediction in predictions:
            yield prediction

def batch(examples):
    batch = []

    for example in examples:
        batch.append(example)

        if len(batch) == batch_size:
            yield batch

            batch = []

    if len(batch) > 0:
        yield batch

def predict_batch(classifier, examples):
    prompts = [example["question"] + example["answer"] for example in examples]
    probabilities = classifier.classify(prompts)

    for example, probability in zip(examples, probabilities):
        example = dict(example)
        example["predictions"] = probability
        yield example

def save_predictions(predictions):
    path = "/app/lamini-classify/data/predictions.jsonl"

    with jsonlines.open(path, "w") as writer:
        for prediction in tqdm(predictions):
            writer.write(prediction)


main()


