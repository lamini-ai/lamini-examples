# Simple Guide to Prompt Engineering

## Introduction

Prompt engineering involves designing prompts that elicit desired response
from the model. You can create prompts for a wide range of purposes, including
information retrieval, problem solving, creative/technical writing,
coding assistance, language translation, opinion generation,
text summerization, conversational agents, data analysis, and more.
It is important to clearly articulate the task or question in the prompt.

In this tutorial, we'll guide you through prompt engineering using lamini's
`MistralRunner()`, which uses the Mistral instruct model by default.

For example, You can write a prompt like
`"What was the decision in Nixon v. United States? Answer in one sentence."`
Lamini allows you to obtain the answer from the Mistral instruct model
with just a few lines of code.

```python
from llama import MistralRunner

runner = MistralRunner()
prompt = "What was the decision in Nixon v. United States? Answer in one sentence."
answer = runner(prompt)
print(answer)
```

It's important to note that there's no need for you to manually include the
special tokens `[INST]` and `[/INST]` around the prompt.
While the Mistral documentation indicates otherwise, our runner will
automatically include these special tokens for you.

Sample output:

```
The decision in Nixon v. United States was that President Richard Nixon was ordered to
release tapes of his conversations with his advisors, as part of a court order to comply
with the Presidential Records Act.
```

Additional prompt examples:
* `How to convert inches to centimeters?`
* `Summarize this story: once upon a time, there was a ...`
* `Write java code for this problem.  Give an array of integers and an integer target, return the indices of the two numbers such that they add up to the target.  Also explain the runtime complexity of the code.`
* `Write a short story about a talking bear who lived in a treehouse.`
* `Translate "hello world, I am a llama" to French.`
* `Suggest innovative solutions to address climate change.`
* `Explain the principles of quantum computing in simple terms.`
* `What would happen if a major historical event had a different outcome?`
* `Envision the world 100 years from now.`
* `If you could time travel to any era, where would you go and why?`

## User vs. System Prompts

A query such as `What was the decision in Nixon v. United States?` represents a user prompt,
seeking specific information and tailored responses, while a system prompt, an optional
directive, sets the context and guides the language model's overall behavior and tone.

You can add a system prompt to generate an output tailored to a legal ethics symposium,
and provide insights relevant to legal professionals and ethicists, like below.

```python
from llama import MistralRunner

runner = MistralRunner()
user_prompt = "What was the decision in Nixon v. United States?"
sys_prompt = "You are a panelist on a legal ethics symposium. Aim to provide a comprehensive analysis suitable for an audience of legal professionals and ethicists."
answer = runner(prompt, system_prompt=sys_prompt)
print(answer)
```
Sample Output:

```
 Nixon v. United States was a landmark case in the United States that involved the President
of the United States, Richard Nixon, and his efforts to cover up illegal activities by his
administration. The case was decided by the Supreme Court of the United States in 1974 and
is considered a significant milestone in the development of legal ethics and the rule of law
in the United States.

In Nixon v. United States, the Supreme Court held that President Nixon had committed a high
crime or misdemeanor in office by covering up illegal activities by his administration,
including the break-in at the Democratic National Committee headquarters during the 1972
presidential campaign. The Court also held that Nixon's actions constituted an abuse of
power and violated his constitutional duty to uphold the law.

The decision in Nixon v. United States was significant for several reasons. First, it
established the principle that no one, not even the President of the United States, is abov
the law and that the rule of law must be upheld regardless of who is in power. Second, the
decision helped to establish the principle of separation of powers, which holds that the
three branches of government - the legislative, executive, and judicial branches -
```

Here are some additional system prompt examples:

* `You are a patient and helpful customer service agent. Help the user troubleshoot.`
* `Imagine you are a poet inspired by nature.`
* `Always assist with care, respect, and truth. Respond with utmost utility yet securely. Avoid harmful, unethical, prejudiced, or negative content. Ensure replies promote fairness and positivity.` (default system prompt in lamini's MistralRunner)

## Mitigate Common Output Issues with Prompt Engineering

### Incomplete Output

It is not uncommon for models to produce incomplete sentences, influenced by various factors such as prompt complexity and training data.
Incomplete sentences could arise due to the model attempting to generate additional context or information.
Refining prompts can help address this behavior.

Consider the output to this prompt ```Discuss the consequences of the industrial revolution.```
Note the output below has an incomplete sentence at the end.

```
The Industrial Revolution was a period of rapid technological and social change that began in the late 18th century and lasted until the mid-19th century. It was marked by the development of new machines and manufacturing processes that revolutionized the way goods were produced and consumed.

One of the most significant consequences of the Industrial Revolution was the growth of the middle class. As more people moved from rural areas to cities to work in factories, there was an increase in the number of people with disposable income to spend on goods and services. This led to the growth of consumer culture and the rise of new forms of entertainment and leisure.

Another consequence of the Industrial Revolution was the rise of urbanization. As more people moved to cities to work in factories, there was a rapid expansion of urban areas. This led to the development of new infrastructure, such as roads, railways, and public transportation systems, to support the growing population.

However, the Industrial Revolution also had its downsides. The working conditions in factories were often dangerous and unhealthy, and many workers were forced to work long hours for low wages. This led to widespread poverty and social unrest, as well as the rise of labor movements and calls
```

To mitigate this issue, We can  update the prompt to
```Discuss the consequences of the industrial revolution in a few sentences.```

Now the output is brief but complete.
```
 The Industrial Revolution was a period of rapid technological advancement and economic growth that began in the late 18th century. It brought about significant changes in the way goods were produced, with the development of machines and factories that made it possible to produce goods on a larger scale and at a faster rate. This led to increased productivity and economic growth, but it also had negative consequences, such as the exploitation of workers and the destruction of traditional ways of life. It is important to consider both the positive and negative aspects of the Industrial Revolution when discussing its impact on society.
 ```

## Iterate and Repeat

The process of refining and improving prompts through successive iterations is cruical
for achieving desired results.  The key is to be adaptable and responsive to the performance of the model and user needs. Here is an example:

* **Observation**
  - **Initial Prompt**: `"Tell me about climate change."`
  - **Observation**: The model provides general information, but the response lacks specificity.
* **Iteration 1** - Refine User Prompt:
  - **Refined Prompt**: `"Explain the impact of human activities on rising global temperatures due to climate change."`
  - **Observation**: The model provides more detailed and focused information.
* **Iteration 2** - Adjust Temperature:
  - **Refined Prompt**: `"Describe the consequences of deforestation on biodiversity and climate change."`
  - **Adjustment**: Lower the temperature for more deterministic responses.
  - **Observation**: The responses become more focused and less varied.
* **Iteration 3** - Test Diverse Scenarios:
  - **Refined Prompt**: `"Discuss the role of renewable energy in mitigating climate change."`
  - **Testing Scenario**: Include prompts related to different renewable energy sources.
  - **Observation**: Evaluate how well the model generalizes across various aspects of the topic.
* **Iteration 4** - Incorporate Feedback.
  - **Refined Prompt**: `"Examine the economic impact of climate change policies on developing nations."`
  - **Feedback**: Users express a desire for insights into economic aspects.
  - **Observation**: The model adapts to provide more information on economic considerations.
* **Iteration 5** - Adjust Max Tokens:
  - **Refined Prompt**: `"Summarize the key findings of the latest IPCC report on climate change."`
  - **Adjustment**: Set a higher max tokens limit for more comprehensive responses.
  - **Observation**: Ensure that longer documents, like summaries, are generated without being cut off.
* **Iteration 6** - Explore Novel Prompts:
  - **Refined Prompt**: `"Imagine you are a journalist reporting on climate change. Provide a news brief highlighting recent developments."`
  - **Observation**: Assess how well the model responds to prompts that simulate real-world scenarios.
* **Iteration 7** - Fine-Tune System Prompts:
  - **Refined System Prompt**: `"You are an expert scientist responding to inquiries about climate change. Maintain a scientific and informative tone."`
  - **Observation**: Evaluate if the refined system prompt influences the model's tone and style.
* **Iteration 8** - Stay Informed:
  - **Refined Prompt**: `"Considering recent advancements, discuss the emerging technologies for carbon capture and their potential impact on mitigating climate change."`
  - **Adjustment**: Incorporate new keywords or concepts based on the latest information.
  - **Observation**: Ensure that the model stays up-to-date with evolving topics.

## Batching Prompts

Batching involves grouping multiple input prompts together and processing them simultaneously as a
batch.  This approach enhances efficiency and speed by allowing the model to handle several prompts
at once, optimizing resource utilization and potentially reducing response time.

In lamini, you can easily batch a list of prompts like below:

```python
from llama import MistralRunner

runner = MistralRunner()
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

## JSON Output with Lamini

While manually crafting prompts for JSON output is possible,
the results may lack consistency and reliability.
Hence, we've introduced a feature to guarantee valid JSON output.

TODO: add example using MistralRunner.

For technical details, see our blog post
[Guarantee Valid JSON Output with Lamini](https://www.lamini.ai/blog/guarantee-valid-json-output-with-lamini).
