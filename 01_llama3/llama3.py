import lamini

#lamini.api_key = "<YOUR-LAMINI-API-KEY>"

llm = lamini.Lamini(model_name="meta-llama/Meta-Llama-3-8B-Instruct")
print(llm.generate("How are you?"))
