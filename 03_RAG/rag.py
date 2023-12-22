from retrieval_augmented_runner import RetrievalAugmentedRunner

llm = RetrievalAugmentedRunner(chunk_size=512, step_size=256)
llm.load_data("data")
llm.train()
prompt = "Have we invested in any generative AI companies in 2023?"
response = llm.call(prompt)
print(response)
