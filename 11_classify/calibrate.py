from sklearn.metrics import PrecisionRecallDisplay, precision_recall_curve
import jsonlines
from argparse import ArgumentParser


def main():
    # Parse command line arguments
    parser = ArgumentParser()

    parser.add_argument(
        "--path",
        type=str,
        default="/app/lamini-classify/data/predictions.jsonl",
        help="path to predictions file",
    )

    args = parser.parse_args()

    examples = load_examples(path=args.path)

    plot_precision_recall_curve(examples)


def load_examples(path):
    with jsonlines.open(path) as reader:
        for example in reader:
            yield example


def plot_precision_recall_curve(examples):
    y_true = []
    y_score = []

    for example in examples:
        if not "predictions" in example:
            continue

        if not "label" in example:
            continue

        y_true.append(example["label"])
        y_score.append(get_positive_class(example["predictions"])["prob"])

    print("plotting precision-recall curve using {} examples".format(len(y_true)))

    display = PrecisionRecallDisplay.from_predictions(y_true, y_score)

    # Save the plot to a file.
    display.figure_.savefig("/app/lamini-classify/data/precision-recall-curve.png")

    precision_target = 0.8

    # Find the threshold for the precision target
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)

    threshold = 0.0

    print("precision: {}".format(precision))
    print("recall: {}".format(recall))
    print("thresholds: {}".format(thresholds))

    for i in range(1, len(precision)):
        if precision[i] >= precision_target:
            threshold = thresholds[i - 1]
            break

    print("threshold for precision target {} is {}".format(precision_target, threshold))


def get_positive_class(predictions):
    for prediction in predictions:
        if prediction["class_name"] == "correct":
            return prediction


main()
