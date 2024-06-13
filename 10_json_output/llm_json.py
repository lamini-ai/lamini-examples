from lamini import Lamini

llm = Lamini(model_name="meta-llama/Meta-Llama-3-8B-Instruct")
response = llm.generate(
    "How old are you?",
    output_type={"age": "int", "units": "str"}
)

print(response)
