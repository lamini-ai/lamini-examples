import lamini

# Instantiate Lamini's LLM client
llm = lamini.Lamini(model_name="meta-llama/Meta-Llama-3.1-8B-Instruct")

# Create a system prompt, test out different prompts such as
#    "You are an expert analyst from Goldman Sachs with 15 years of experience."
#    or "You are an influencer who loves emojis."
system_prompt = "You are a helpful assistant."

# Think of a question
question = "How much is a carton of pasture raised eggs?"

# Form the prompt using the Llama 3.1 prompt template
prompt = f"""\
<|begin_of_text|><|start_header_id|>system<|end_header_id|>

{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>

{question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""

# Send the prompt to the LLM to generate an answer!
response = llm.generate(prompt)

print(f"Prompt: {prompt}\nResponse: {response}")
