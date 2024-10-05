# Prompt Tuning

Consider watching this video explaining how to prompt tune Open LLMs: https://www.youtube.com/watch?v=f32dc5M2Mn0

Run Llama 3 on a few example questions:

```bash
cd 03_prompt_tuning
python3 generate.py
```

Try editing the prompt and see what happens! For example, try adding `"You are an expert analyst from Goldman Sachs with 15 years of experience."` or `"You are an influencer who loves emojis."`

## Guidelines

### Iterate quickly

Try out many prompts quickly instead of thinking hard about the perfect prompt. Good prompt engineers can try about 100 different prompts in an hour.  If you are spending more than 1 hour on prompt tuning, you should move on.

### Don't forget the template

This code adds the prompt template for Llama 3.1. Don't forget it! The model will perform much worse without the correct template. Every model has a different template. Look it up on the model card, e.g. [Llama3.1 model card](https://www.llama.com/docs/model-cards-and-prompt-formats/llama3_1/).

```python
<|begin_of_text|><|start_header_id|>system<|end_header_id|>

{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>

{user_input}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
```

### Integrate data sources

Plug relevant information from your relational database, knowledge graph, recommendation system, etc into your prompt.

E.g. if you are building Q&A bot that answers questions about the document the user is viewing, pull the document title & summary from a database and insert it into the prompt.

### Deeper dive

Want to learn more? We have even more details about prompt tuning: [Prompt Engineering](prompt_engineering.md).
