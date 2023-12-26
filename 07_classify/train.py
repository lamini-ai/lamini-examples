from lamini_classifier import LaminiClassifier
import jsonlines

import logging

def main():
    #logging.basicConfig(level=logging.DEBUG)

    classifier = LaminiClassifier(augmented_example_count=32)

    examples = load_examples()

    add_data(classifier, examples)

    prompts={
      "incorrect": "Questions with incorrect answers.  Imagine that you are a medical expert reading the question and answers about the ICD11 standard.",
      "correct": "Questions with correct answers.  Imagine that you are a medical expert reading the question and answers about the ICD11 standard."
    }

    classifier.prompt_train(prompts)

    classifier.save("models/classifier.lamini")

def load_examples():
    path = "/app/lamini-classify/data/questions_and_answers.jsonl"

    with jsonlines.open(path) as reader:
        for example in reader:
            yield example

def add_data(classifier, examples):
    positive_examples = []
    negative_examples = []
    for example in examples:
        if "label" in example:
            if example["label"] == 1:
                positive_examples.append(example["question"] + " " + example["answer"])
            else:
                negative_examples.append(example["question"] + " " + example["answer"])

    if len(positive_examples) > 0:
        classifier.add_data_to_class("correct", positive_examples)

    if len(negative_examples) > 0:
        classifier.add_data_to_class("incorrect", negative_examples)

main()
