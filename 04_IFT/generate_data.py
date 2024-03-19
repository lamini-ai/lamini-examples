import random
import jsonlines
import os

from tqdm import tqdm
from lamini import MistralRunner
import lamini

batch_size = 50
max_questions = 50000

lamini.max_workers = 40


def main():
    random.seed(42)

    entities = load_entities(
        path="/app/lamini-ift/data/best_entities_with_descriptions.jsonl"
    )

    questions = generate_questions(entities)

    questions_and_answers = generate_answers(questions)

    save_questions_and_answers(questions_and_answers)


def load_entities(path):
    print("Loading entities from", path)
    with jsonlines.open(path) as reader:
        entities = []
        for obj in tqdm(reader):
            entities.append(obj)
        random.shuffle(entities)

    return entities


def generate_questions(entities):
    entity_batches = batch(entities, batch_size)

    for entity_batch in entity_batches:
        question_batch = QuestionGenerator(entity_batch)

        for question in question_batch:
            yield question


def batch(iterable, n=1):
    batch = []
    for item in iterable:
        batch.append(item)
        if len(batch) == n:
            yield batch
            batch = []

    if len(batch) > 0:
        yield batch


class QuestionGenerator:
    def __init__(self, entities):
        self.entities = entities
        self.questions = None
        self.error = None

    def __iter__(self):
        for index in range(3 * len(self.entities)):
            yield Question(self, index)

    def get(self):
        if self.error is not None:
            raise self.error

        if self.questions is None:
            self.questions = self.generate_questions()

        return self.questions

    def generate_questions(self):
        prompt_batch = [self.generate_prompt(entity) for entity in self.entities]

        runner = MistralRunner()

        try:
            questions_batch = runner(
                prompt_batch,
                system_prompt=self.get_system_prompt(),
                output_type={"question1": "str", "question2": "str", "question3": "str"}
            )

        except Exception as e:
            self.error = e
            raise e

        questions = []

        for entity, questions_dict in zip(self.entities, questions_batch):
            questions.append(questions_dict["question1"])
            questions.append(questions_dict["question2"])
            questions.append(questions_dict["question3"])

        return questions

    def generate_prompt(self, entity):
        prompt = "You are going to read through a page from the standard carefully and generate three questions about it."
        prompt += entity["description"]

        return prompt

    def get_system_prompt(self):
        return "You are a medical coding expert who has just read the recent ICD11 standard from the World Health Organization."


class Question:
    def __init__(self, question_generator, index):
        self.question_generator = question_generator
        self.index = index

    def get(self):
        question = self.question_generator.get()[self.index]
        entity = self.question_generator.entities[self.index // 3]

        return {"entity": entity, "question": question}


def generate_answers(questions):
    question_batches = batch(questions, batch_size)

    for question_batch in question_batches:
        answer_batch = AnswerGenerator(question_batch)

        for answer in answer_batch:
            yield answer


class AnswerGenerator:
    def __init__(self, questions):
        self.questions = questions
        self.answers = None
        self.error = None

    def __iter__(self):
        for index, question in enumerate(self.questions):
            yield Answer(self, index)

    def get(self):
        if self.error is not None:
            raise self.error

        if self.answers is None:
            self.answers = self.generate_answers()

        return self.answers

    def generate_answers(self):
        prompt_batch = [self.generate_prompt(question) for question in self.questions]

        runner = MistralRunner()

        try:
            answers_batch = runner(
                prompt_batch,
                system_prompt=self.get_system_prompt()
            )
        except Exception as e:
            self.error = e
            raise e

        return answers_batch

    def generate_prompt(self, question):
        prompt = "You are going to read through a page from the standard carefully, and answer a question about it.\n\n"
        prompt += question.get()["entity"]["description"] + "\n\n"
        prompt += "Now answer the following question:\n\n"
        prompt += question.get()["question"]

        return prompt

    def get_system_prompt(self):
        return "You are a medical coding expert who has just read the recent ICD11 standard from the World Health Organization."


class Answer:
    def __init__(self, answer_generator, index):
        self.answer_generator = answer_generator
        self.index = index

    def get(self):
        answer = self.answer_generator.get()[self.index]
        question = self.answer_generator.questions[self.index].get()

        return {
            "question": question["question"],
            "entity": question["entity"],
            "answer": answer,
        }


def save_questions_and_answers(questions_and_answers):
    print("Generating up to {} questions and answers".format(max_questions))

    path = "/app/lamini-ift/data/questions_and_answers.jsonl"

    # Count the number of json objects in the file
    if os.path.exists(path):
        with jsonlines.open(path) as reader:
            num_objects = sum(1 for obj in reader)
    else:
        num_objects = 0

    print("Fast forwarding to object number", num_objects)

    with jsonlines.open(path, mode="a") as writer:
        index = 0
        for question_and_answer in tqdm(questions_and_answers):
            # Skip the objects that have already been written to the file
            if index < num_objects:
                index += 1
                continue

            if index >= max_questions:
                break

            try:
                # Get performs the computation
                row = question_and_answer.get()

                writer.write(row)

                index += 1
            except Exception as e:
                print("Error generating question and answer", e)


main()
