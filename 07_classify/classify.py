
from lamini_classifier import LaminiClassifier

classifier = LaminiClassifier()

classifier.prompt_train()
prompts={
  "correct": "Questions with correct answers.",
  "incorrect": "Questions with incorrect answers."
}

llm.prompt_train(prompts)

llm.save("models/my_model.lamini")


