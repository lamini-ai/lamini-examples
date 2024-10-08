import faiss
import lamini
import jsonlines

# Number of nearest chunks to return
k = 2

# Set up for the index, which holds the embeddings, and the splits, which holds the corresponding plain text
index = None
splits = []

# Instantiate Lamini's embedding client
embedding_client = lamini.Embedding()

with jsonlines.open("data.jsonl", "r") as file:
    for item in file:
        # Create an embedding for every 'transcript' item in the file and add to the index
        transcript_embedding = embedding_client.generate(item['transcript'])
        if not index:
            index = faiss.IndexFlatL2(transcript_embedding.size) # Set the size of the index based on model embedding size
        index.add(transcript_embedding)
        splits.append(item['transcript'])

question = "What is TSMC's 2019 revenue in USD?"

# Generate the embedding for the question
question_embedding = embedding_client.generate(question)

# Find the k nearest neighbors in the index for the question embedding
distances, indices = index.search(question_embedding, k)

# Retrieve the relevant data from the splits based on hits in the index
relevant_data = [splits[i] for i in indices[0] if i >= 0]

# Instantiate Lamini's LLM client
llm = lamini.Lamini(model_name="meta-llama/Meta-Llama-3.1-8B-Instruct")

# Form the prompt using the RAG hits and the question
prompt = f"""\
<|begin_of_text|><|start_header_id|>user<|end_header_id|>

{relevant_data}
{question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""

print(prompt)

# Send the prompt to the LLM to generate an answer!
response = llm.generate(prompt)
print(response)
