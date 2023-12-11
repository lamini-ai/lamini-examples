from llama import RetrievalAugmentedRunner

llm = RetrievalAugmentedRunner(chunk_size=512, step_size=256, k=5)
llm.load_data("data")
llm.train()
prompt = "Have we invested in any generative AI companies in 2023?"
response = llm(prompt)
print(response)
