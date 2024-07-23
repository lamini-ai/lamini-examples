from lamini import LaminiClassifier

llm = LaminiClassifier()

prompts={
  "correct": "Questions with correct answers.",
  "incorrect": "Questions with incorrect answers."
}

llm.prompt_train(prompts)

llm.save("my_model.lamini")