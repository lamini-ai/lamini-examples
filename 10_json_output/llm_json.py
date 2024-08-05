from lamini import Lamini

# Instatiate Lamini object to interface with LLMs on the compute server
llm = Lamini(model_name="meta-llama/Meta-Llama-3-8B-Instruct")

# Provide query to generate and provide output type as a dictionary
response = llm.generate(
    "How old are you?",
    output_type={"age": "int", "units": "str"}
)

# Output response
print(response)
