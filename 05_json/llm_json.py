from lamini import Lamini

llm = Lamini(model_name="meta-llama/Llama-2-7b-chat-hf")
result = llm.generate(
    "What is the ICD11 code for swelling of the liver?",
    output_type={"icd11_code": "str", "explanation": "str"}
)

print(result)

