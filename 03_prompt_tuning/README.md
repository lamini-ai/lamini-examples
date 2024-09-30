<div align="center">
<img src="https://avatars.githubusercontent.com/u/130713213?s=200&v=4" width="110"><img src="https://huggingface.co/lamini/instruct-peft-tuned-12b/resolve/main/Lamini_logo.png?max-height=110" height="110">
</div>

# Prompt Tuning

Consider watching this video explaining how to prompt tune Open LLMs: https://www.youtube.com/watch?v=f32dc5M2Mn0

Run Llama 3 on a few example questions:

```bash
./scripts/prompt_tune.sh
```

You can view the results in [data/results/spot_check_results.jsonl](data/results/spot_check_results.jsonl).

# Editing the prompt

Quicky iterate on different prompts by editing the [Prompt code](spot_check.py#L94), and running the spot check.

https://github.com/lamini-ai/lamini-examples/blob/d01af0bcd91d135098f4e099f82b24b44f52d414/03_prompt_tuning/spot_check.py#L93-L109


For example, try changing `"You are an expert analyst from Goldman Sachs with 15 years of experience."` to `"You are an influencer who loves emojis."` and see what happens!

# Guidelines

## Iterate quickly

Try out many prompts quickly instead of thinking hard about the perfect prompt. Good prompt engineers can try about 100 different prompts in an hour.  If you are spending more than 1 hour on prompt tuning, you should move on.

## Don't forget the template

This code adds the prompt template for Llama 3.1. Don't forget it! The model will perform much worse without the correct template. Every model has a different template. Look it up on the model card, e.g. [Llama3.1 model card](https://www.llama.com/docs/model-cards-and-prompt-formats/llama3_1/).

```python
async def add_template(self, prompts):
    async for prompt in prompts:

        new_prompt = "<|begin_of_text|><|start_header_id|>user<|end_header_id|>"
        new_prompt += prompt.data.get_prompt() + "<|eot_id|>"
        new_prompt += "<|start_header_id|>assistant<|end_header_id|>"
```

## Integrate data sources

Plug relevant information from your relational database, knowledge graph, recommendation system, etc into your prompt.

E.g. if you are building Q&A bot that answers questions about the document the user is viewing, pull the document title & summary from a database and insert it into the prompt.

# Deeper dive
Want to learn more? We have even more details about prompt tuning: [Prompt Engineering](prompt_engineering.md).
