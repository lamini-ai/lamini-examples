import faiss
import lamini
import numpy as np
import jsonlines

EMBEDDING_SIZE = 384
k = 2

index = faiss.IndexFlatL2(EMBEDDING_SIZE)
splits = []

embedding_client = lamini.Embedding()

with jsonlines.open("data.jsonl", "r") as file:
    for item in file:
        transcript_embedding = embedding_client.generate(item['transcript'])
        index.add(transcript_embedding)
        splits.append(item['transcript'])

question = "What is TSMC's 2019 revenue in USD?"

question_embedding = embedding_client.generate(question)
distances, indices = index.search(question_embedding, k)
relevant_data = [splits[i] for i in indices[0] if i >= 0]

llm = lamini.Lamini(model_name="meta-llama/Meta-Llama-3.1-8B-Instruct")
prompt = f"""\
<|begin_of_text|><|start_header_id|>user<|end_header_id|>

{relevant_data}
{question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""

print(prompt)

response = llm.generate(prompt)
print(response)
