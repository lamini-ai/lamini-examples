import lamini

# Uncomment the statement below and insert your API key from
# https://app.lamini.ai/account
# lamini.api_key = "<YOUR-LAMINI-API-KEY>"

llm = lamini.Lamini(model_name="meta-llama/Meta-Llama-3.1-8B-Instruct")
print(llm.generate("How are you?", output_type={"Response":"str"}))
