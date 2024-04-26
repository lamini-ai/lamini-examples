from lamini import Lamini

import argparse
import random
import os
import jsonlines

from tqdm import tqdm

import logging

logger = logging.getLogger(__name__)


def main():
    """Compare a set of answers to a set of questions."""
    parser = argparse.ArgumentParser(
        description="Compare a set of answers to a set of questions."
    )

    # The input to the program is a jsonlines file containing the golden reference answers
    parser.add_argument(
        "--golden-input",
        nargs="?",
        help="The path to the golden input questions and answers.",
        default="/app/lamini-eval/data/gold/answers.jsonl",
    )

    # The input to the program is a jsonlines file containing the model-a answers
    parser.add_argument(
        "--model-a",
        nargs="?",
        help="The path to the model-a input questions and answers.",
        default="/app/lamini-eval/data/model-a/answers.jsonl",
    )

    # The input of the program is a jsonlines file containing the model-b answers
    parser.add_argument(
        "--model-b",
        nargs="?",
        help="The path to the model-b input questions and answers.",
        default="/app/lamini-eval/data/model-b/answers.jsonl",
    )

    # The output of the program is a jsonlines file containing the comparisons
    parser.add_argument(
        "--output-answers",
        nargs="?",
        help="The path to the output questions and answers.",
        default="/app/lamini-eval/data/comparisons.jsonl",
    )

    # The batch size
    parser.add_argument(
        "--batch-size",
        help="The batch size to use.",
        default=8,
        type=int,
    )

    logging.basicConfig(level=logging.INFO)

    # Parse the arguments
    args = parser.parse_args()

    golden_answers = load_answers(args.golden_input)
    model_a_answers = load_answers(args.model_a)
    model_b_answers = load_answers(args.model_b)

    # Run the model to compare the answers
    comparisons = run_model(
        golden_answers, model_a_answers, model_b_answers, args.batch_size
    )

    # Save the comparisons
    save_comparisons(args.output_answers, comparisons)


def load_answers(path):
    """Load the answers."""
    with jsonlines.open(path) as reader:
        answers = [answer for answer in reader]
    return answers


def run_model(golden_answers, model_a_answers, model_b_answers, batch_size):
    """Run the model to compare the answers."""
    llm = Lamini()

    # Form a prompt for each pair of answers
    questions = []
    for golden_answer, model_a_answer, model_b_answer in zip(golden_answers, model_a_answers, model_b_answers):
        question = "Two models (A and B) are going to answer the same question about the WHO ICD11 standard. Your job is to rate their answers, comparing them to a golden reference.  You are an expert rater.\n\n"
        question += (
            f"Rate the answers using a similarity scale from 1 (lowest similarity) to 5 (highest similarity). Use the full range. Read the gold answer carefully. Prefer answers that are most similar to the gold answer, even if the gold answer refused to answer the question.\n\n"
        )
        question += f"========== question =========\n{model_a_answer['question']}\n\n"
        question += f"========== gold answer =========\n{golden_answer['answer']}\n\n"
        question += f"========== model A answer =========\n{model_a_answer['answer']}\n\n"
        question += f"========== model B answer==========\n{model_b_answer['answer']}\n\n"
        question += "=" * 40 + "\n\n"
        question += f"Which model most similar to the gold answer, A or B?\n\n"
        questions.append(question)

    # Batch the questions
    question_batches = batch_questions(questions, batch_size)

    average_score = 0.0
    total_questions = 0

    for question_batch in question_batches:
        answers_batch = llm.generate(
            question_batch,
            output_type={"model_a_similarity": "int", "model_b_similarity": "int"},
        )

        for question, answer in zip(question_batch, answers_batch):
            # The normalized score is the difference between the model-b rating and the model-a rating
            # It starts at 0.0 and goes up to 1.0.  When it is 0.5, the models are equally good.
            normalized_score = (
                answer["model_b_similarity"] - answer["model_a_similarity"]
            ) / 8.0 + 0.5
            average_score = (
                average_score * total_questions + normalized_score
            ) / (total_questions + 1)


            question_and_answer = {
                "question": question,
                "model_a_similarity": answer["model_a_similarity"],
                "model_b_similarity": answer["model_b_similarity"],
                "normalized_score": normalized_score,
            }

            yield question_and_answer

            total_questions += 1

    logger.info(f"Average Score: {average_score}")


def batch_questions(questions, batch_size):
    """Batch the questions."""
    batches = []
    for i in range(0, len(questions), batch_size):
        batches.append(questions[i : i + batch_size])
    return batches


def save_comparisons(path, comparisons):
    """Save the answers."""
    with jsonlines.open(path, "w") as writer:
        for comparison in tqdm(comparisons):
            logger.info("=========================================")
            logger.info(f"Question: {comparison['question']}")
            logger.info(f"Model A Rating: {comparison['model_a_similarity']}")
            logger.info(f"Model B Rating: {comparison['model_b_similarity']}")
            logger.info(f"Normalized Score: {comparison['normalized_score']}")
            logger.info("=========================================")
            writer.write(comparison)


main()
