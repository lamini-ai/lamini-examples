from retrieval_augmented_runner import RetrievalAugmentedRunner

# For more verbose output, set the logging level to INFO
import logging
logging.basicConfig(level=logging.INFO)

# Create a runner and load the data
runner = RetrievalAugmentedRunner(chunk_size=512, step_size=256)
runner.load_data("data")

# Generate and save the index
runner.train()
runner.index.save_index("index")

# Ask a question with RAG
prompt = "Have we invested in any generative AI companies in 2023?"
response = runner.call(prompt)
print(response)

# Try changing the number of RAG chunks, notice that the single RAG hit isn't enough for the model to answer the question
runner.k = 1
response = runner.call(prompt)
print(response)

# Try prompt-tuning the model!
prompt = "You are a analyst at an investment company that knows everything. Have we invested in any generative AI companies in 2023?"
response = runner.call(prompt)
print(response)
