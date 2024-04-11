import lamini
import jsonlines
from tqdm import tqdm

def main():
    questions = load_questions()

    answers = answer_questions(questions)

    save_answers(answers)

def load_questions():
    with jsonlines.open("sample_questions.jsonl") as reader:
        questions = [q for q in reader]

    return questions

def answer_questions(questions):
    answers = []

    for question in tqdm(questions):
        llm = lamini.MistralRunner()

        answer = llm(question["question"])

        answers.append({
            "question": question["question"],
            "answer": answer
        })

    return answers

def save_answers(answers):
    with jsonlines.open("data/sample_answers.jsonl", mode="w") as writer:
        writer.write_all(answers)

if __name__ == "__main__":
    main()
