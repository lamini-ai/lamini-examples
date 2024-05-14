from lamini import Lamini

llm = Lamini(model_name="meta-llama/Meta-Llama-3-8B-Instruct")
llm.generate(
    "How old are you?",
    output_type={"age": "int", "units": "str"}
)

