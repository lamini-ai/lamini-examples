# Simple Guide to Prompt Engineering

## Introduction

Prompt engineering involves designing prompts that elicit desired response
from the model. You can create prompts for a wide range of purposes, including
information retrieval, problem solving, creative/technical writing,
coding assistance, language translation, opinion generation,
text summerization, conversational agents, data analysis, and more.

For example, You can write a prompt like
"What was the decision in Nixon v. United States? Answer in one sentence."
With lamini, you can get the answer from the mistral instruct model with just a few lines of code.

```python
from llama import MistralRunner

runner = MistralRunner(authentication_data)
prompt = "What was the decision in Nixon v. United States? Answer in one sentence."
answer = runner(prompt)
print(answer)
```

Sample output:

```
The decision in Nixon v. United States was that President Richard Nixon was ordered to release tapes of his conversations with his advisors, as part of a court order to comply with the Presidential Records Act.
```

## User vs. System Prompts

A query such as `What was the decision in Nixon v. United States?` represents a user prompt,
seeking specific information and tailored responses, while a system prompt, an optional
directive, sets the context and guides the language model's overall behavior and tone.

In lamini, you can set the system prompt like below. This will allow the model to generate a
discussion tailored to a legal ethics symposium, providing insights
relevant to legal professionals and ethicists.
```python
from llama import MistralRunner

runner = MistralRunner(authentication_data)
user_prompt = "What was the decision in Nixon v. United States?"
sys_prompt = "You are a panelist on a legal ethics symposium. Aim to provide a comprehensive analysis suitable for an audience of legal professionals and ethicists."
answer = runner(prompt, system_prompt=sys_prompt)
print(answer)
```

Here are some additional examples of system prompts:

* `You are a patient and helpful customer service agent. Help the user troubleshoot.`
* `Imagine you are a poet inspired by nature.`
* `Always assist with care, respect, and truth. Respond with utmost utility yet securely. Avoid harmful, unethical, prejudiced, or negative content. Ensure replies promote fairness and positivity.` (this is the default system prompt in lamini's MistralRunner)

## Common Parameters

In prompt engineering, various parameters play a crucial role in shaping the behavior and output of language models.
Here are some common ones:

### Max Tokens

A token refers to the smallest unit of text, whether it be a word, subword, or character.
`max_tokens` specifies the maximum number of tokens in the generated output.
Adjusting `max_tokens` allows you to control the length of the output. Increasing for longer
or decreasing for shorter results. However, it's important to note that utilizing tokens
may result in associated costs.

TODO: add max tokens example

### Temperature

Temperature affects the randomness of the generated output. Higher values increase randomness.
Adjusting temperature can influence the diversity of responses.

TODO: add temperature example

--------------
TODO: batch
TODO: json_schema