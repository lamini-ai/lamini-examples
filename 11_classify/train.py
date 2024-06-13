from lamini_classifier import LaminiClassifier

import csv
import jsonlines

import logging

def main():
    #logging.basicConfig(level=logging.DEBUG)

    classifier = LaminiClassifier(augmented_example_count=8)

    examples = load_examples()

    add_data(classifier, examples)

    prompts={
      "incorrect": "Questions with incorrect answers. Imagine you are a legislative analyst reading a bill.",
      "correct": "Questions with correct answers. Imagine you are a legislative analyst reading a bill."
    }

    classifier.prompt_train(prompts)

    classifier.save("models/classifier.lamini")

def load_examples_2():
    path = "/app/lamini-classify/data/qa_pairs__2024_01_13__05_47_59.csv"

    return read_csv_as_list_of_dicts(path)

def read_csv_as_list_of_dicts(path):
    rows = []
    with open(path) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=",")
        first_line = True
        headers = None
        for row in csv_reader:
            if first_line:
                headers = row
                first_line = False
                continue
            else:
                dictt = {}
                for idx, column_data in enumerate(row):
                    dictt[headers[idx]] = column_data

    return rows

def load_examples():
    path = "/app/lamini-classify/data/predictions.jsonl"

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
