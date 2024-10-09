import lamini

lamini.gate_pipeline_batch_completions = True

# Instatiate Classifier object to pass prompts to the LLM on the compute server
llm = lamini.LaminiClassifier()

# Build prompt dictionary which holds the classes as the keys (correct, incorrect)
# with the values provided descriptions for what defines the class.
prompts={
  "correct": "Questions with correct answers.",
  "incorrect": "Questions with incorrect answers."
}

# Call prompt train to train the LLM with the provided classes and prompts
if llm.prompt_train(prompts):

  # Save the model for later use
  llm.save("my_model.lamini")
