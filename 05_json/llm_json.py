from lamini import Lamini

llm = Lamini(model_name="meta-llama/Llama-2-7b-chat-hf")
llm.generate(
    "How old are you?",
    output_type={"age": "int", "units": "str"}
)

