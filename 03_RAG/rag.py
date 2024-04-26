from retrieval_augmented_runner import RetrievalAugmentedRunner

# For more verbose output, set the logging level to INFO
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create a RAG runner and load the data
rag = RetrievalAugmentedRunner(chunk_size=512, step_size=128)
rag.load_data("data")

# Text splitting
# Text embedding
# Generate and save the index in Vector stores
rag.train()
rag.index.save_index("index")

# Ask a question with RAG
prompt = "Have we invested in any generative AI companies in 2023?"
most_similar = rag.query(prompt)
augmented_prompt = "\n".join(reversed(most_similar)) + "\n\n" + prompt
response = rag.generate(augmented_prompt)
logger.info(response)

# Try changing the number of RAG chunks, notice that the single RAG hit isn't enough for the model to answer the question
rag.k = 1
most_similar = rag.query(prompt)
augmented_prompt = "\n".join(reversed(most_similar)) + "\n\n" + prompt
response = rag.generate(augmented_prompt)
logger.info(response)

# Try prompt-tuning the model!
prompt = "You are a analyst at an investment company that knows everything. Have we invested in any generative AI companies in 2023?"
most_similar = rag.query(prompt)
augmented_prompt = "\n".join(reversed(most_similar)) + "\n\n" + prompt
response = rag.generate(augmented_prompt)
logger.info(response)
