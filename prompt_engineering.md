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

Temperature affects the randomness of the generated output.
* Temperature 0: The model selects the most probable token at each step. This leads to more focused and deterministic responses.
* Low Temperature (Close to 0): Results in more deterministic and conservative outputs, with a focus on the most probable tokens.
* Moderate Temperature (Around 1): Allowing for a mix of likely and less likely tokens in the output.
* High Temperature (Above 1): Introduces more randomness and diversity into the generation process, leading to more creative and varied responses.

TODO: add temperature example
TODO: check if temperature must be float

## Batching

Batching involves grouping multiple input prompts together and processing them simultaneously as a
batch.  This approach enhances efficiency and speed by allowing the model to handle several prompts
at once, optimizing resource utilization and potentially reducing response time.

In lamini, you can easily batch a list of prompts like below:

```python
from llama import MistralRunner

runner = MistralRunner(authentication_data)
prompts = ["Is pizza nutritous?",
           "Did Richard Nixon reisgn?",
           "Summarize the impact of global warming.",
          ]
answer = runner(prompts, system_prompt="Provide very short responses."))
print(answer)
```

Sample output:
```
[{'input': 'Is pizza nutritous?',
  'output': 'No, pizza is not typically considered a nutritious food due to its high calorie, carbohydrate, and fat content. However, it can be made healthier by using whole grain crust, lean protein toppings, and plenty of vegetables.'},
 {'input': 'Did Richard Nixon reisgn?',
  'output': ' Yes, Richard Nixon resigned as President of the United States on August 9, 1974.'},
 {'input: 'Summarize the impact of global warming.',
  'output': " Global warming has significant impacts on the Earth's environment, including rising sea levels, more frequent and intense heatwaves, droughts, and extreme weather events. It also affects wildlife, agriculture, and human health. The main cause of global warming is the increase in greenhouse gases in the atmosphere, primarily from human activities such as burning fossil fuels and deforestation. Addressing global warming requires reducing greenhouse gas emissions and transitioning to renewable energy sources."}]
```


--------------
TODO: batch
TODO: json_schema