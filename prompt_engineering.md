# Simple Guide to Prompt Engineering :rocket:

## Introduction

Prompt engineering involves designing prompts or inputs that elicit desired response
from the model.
You can create prompts for a wide range of purposes, including
information retrieval, problem solving, creative/technical writing,
coding assistance, language translation, opinion generation,
text summerization, conversational agents, data analysis, and more.

Here are some simple prompt examples:

* `How to convert inches to centimeters?`
* `Summarize this story: once upon a time, there was a ...`
* `Write java code for this problem with optimal runtime complexity.  Give an array of integers and an integer target, return the indices of the two numbers such that they add up to the target.  Also explain the runtime complexity of the code.`
* `Write a short story about a talking bear who lived in a treehouse.`
* `Translate "hello world, I am a llama" to French.`
* `Suggest innovative solutions to address climate change.`
* `Explain the principles of quantum computing in simple terms.`
* `What would happen if WII had a different outcome?`
* `Envision the world 100 years from now.`

It is very important to clearly articulate the task or question in the prompt.
In addition, iterating and refining prompts is crucial for achieving optimal
results and harnessing the full potential of a language model.

In this tutorial, we'll guide you through prompt engineering using Lamini's
`MistralRunner`, which uses the Mistral instruct model by default and allows
you to obtain the response with just a few lines of code, like below.

```python
from llama import MistralRunner

runner = MistralRunner()
prompt = "What was the decision in Nixon v. United States? Answer in one sentence."
answer = runner(prompt)
print(answer)
```

To prompt the Mistral instruct model effectively and get optimal responses,
it is recommended to wrap
`[INST]` and `[/INST]` around the prompt as shown below.
However, our runner will automatically wrap the prompt before passing it to the model.
```
"[INST]What was the decision in Nixon v. United States? Answer in one sentence.[/INST]"
```

Output:

```
The decision in Nixon v. United States was that President Richard Nixon was ordered to
release tapes of his conversations with his advisors, as part of a court order to comply
with the Presidential Records Act.
```

Although the response is factual, you may want the response to be phrased differently.
Perhaps you are looking for a more elaborate response?
Perhaps you prefer the response tailored for a particular audience or have other constraints?
Follow this tutorial to learn how to iterate and refine your prompts to generate excellent responses. :rocket:

## User vs. System Prompts

A query such as `"What was the decision in Nixon v. United States?"` represents a user prompt,
seeking specific information and tailored responses.
On the other hand, a system prompt, an optional
directive, sets the context and guides the language model's overall behavior and tone.

For example, you can add a system prompt to generate an output tailored to a legal ethics symposium,
and provide insights relevant to legal professionals and ethicists, like below.

```python
from llama import MistralRunner

runner = MistralRunner()
user_prompt = "What was the decision in Nixon v. United States?"
sys_prompt = "You are a panelist on a legal ethics symposium. Aim to provide a comprehensive analysis suitable for an audience of legal professionals and ethicists."
answer = runner(prompt, system_prompt=sys_prompt)
print(answer)
```
Output:

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

Additional system prompt examples:

* `You are a patient and helpful customer service agent. Help the user troubleshoot.`
* `Imagine you are a poet inspired by nature.`
* `Always assist with care, respect, and truth. Respond with utmost utility yet securely. Avoid harmful, unethical, prejudiced, or negative content. Ensure replies promote fairness and positivity.` (default system prompt in Lamini's MistralRunner)

## Refining Prompts

In prompt engineering, you can manipulate different attributes of the responses.
Here are some key attributes you can control:

### Response Length

It is not uncommon for a model to produce sentences that are too long or incomplete, influenced by various factors such as prompt complexity and training data.
Similarly, the model may produce responses shorter than your desired length.

Here are a few techniques to get the desired output lengths.
* Specify desired length:
  - Explicitly state phrases like `"Generate a response with up to two paragraphs."` or `"End reponse after 50 words."`
* Use keywords and constraints:
  - Include keywords or constraints like `"brief"`, `"succinct"` or `"Write a detailed paragraph."`.

Consider the output to this prompt ```"Describe the impacts of Silicon Valley."```
It is long and has an incomplete sentence at the end.

```
 Silicon Valley is a region in California that is known for its high-tech industry and innovation. It is home to some of the world's largest and most successful technology companies, including Google, Facebook, Apple, and Microsoft. The region has had a significant impact on the global economy and has transformed the way we live, work, and communicate.

One of the most significant impacts of Silicon Valley is the creation of new jobs and industries. The region has become a hub for innovation and entrepreneurship, attracting talented individuals from around the world who are looking to work on cutting-edge technologies. This has led to the creation of new industries such as artificial intelligence, virtual reality, and blockchain, which have the potential to revolutionize many aspects of our lives.

Silicon Valley has also had a significant impact on the global economy. The region is home to some of the world's largest and most successful technology companies, which have created billions of dollars in revenue and have transformed the way we do business. This has led to increased competition and innovation in many industries, which has driven down costs and increased efficiency.

However, Silicon Valley has also been criticized for its impact on society. Some have argued that the region's
```

The prompt below produces a response that is brief and complete.
```
Describe the impacts of Silicon Valley in a few sentences.
```

Output:
```
 Silicon Valley has had a significant impact on the world, driving innovation and technological advancements that have transformed industries and improved our lives in countless ways. The region is home to some of the world's largest and most influential tech companies, including Google, Facebook, and Apple, which have created millions of jobs and contributed billions of dollars to the global economy. However, Silicon Valley has also faced criticism for its impact on society, including concerns about privacy, social isolation, and the widening wealth gap. Overall, the region represents both incredible potential and challenges for the future.
 ```

We can get an even shorter response with
```
Describe the impacts of Silicon Valley in a few words.
```

Output:
```
 Silicon Valley has revolutionized technology, created jobs, and transformed industries.
```

### Output Format

You can guide the model to generate responses in the desired format,
whether it be a list, a table, or a customized format.  Here are some key strategies:

* Instructional Clarity - use explicit language to instruct the model on the desired format.
  - Ex. Use phrases like `"Generate the response as a list"` or `"Present the information in table format"`.
* Example Illustration - include a clear example of the desired output format within your prompt. Show a sample list or table and instruct the model to follow that structure, helping it understand your expectations.
  - Ex.
    ```
    Describe the advantages of renewable energy. Provide the response in a bulleted list format. For instance:
    Environmental Sustainability: Decreases carbon footprint and minimizes environmental impact.
    Cost-Efficiency: Long-term savings through reduced reliance on fossil fuels.
    Energy Independence: Reduces dependence on non-renewable resources.

    Follow a similar structure in your response.
    ```

### Creativity or Precision Level

You can control your desired level of creativity or precision using alternative phrasing.
Here are some examples:

* Higher Creativity
  - `"Generate a narrative with a more exploratory tone."`
  - `"Compose a story with a touch of unpredictability."`
  - `"Provide a response that allows for a broader range of possibilities."`
* Lower Precision
  - `"Deliver a straightforward and concise explanation."`
  - `"Offer a focused and to-the-point response."`
  - `"Provide information with a higher level of certainty."`
* Experimentation
  - `"Explore various perspectives in your response."`
  - `"Try different approaches in your explanation."`
  - `"Adjust your writing style to see how it affects the output."`

### Rephrase

Consider this response

```
The decision in Nixon v. United States was that President Richard Nixon was ordered to release tapes of his conversations with his advisors, as part of a court order to comply with the Presidential Records Act.
```

We can ask the model to rephrase this response in the prompt.
```
Rewrite this: The decision in Nixon v. United States was that President Richard Nixon was ordered to release tapes of his conversations with his advisors, as part of a court order to comply with the Presidential Records Act.
```

Output:
```
 The Nixon v. United States case resulted in a court order requiring President Richard Nixon to disclose recordings of his discussions with advisors, in accordance with the Presidential Records Act.
```

### Where to Place Constraints?

Placing instructions or constraints like `"Write in one sentence."` at the end
of the prompt is often recommended.
Giving the model the chance to understand the context before encountering constraints can generally lead to more contextually appropriate and coherent responses.
On the other hand, adding constraints to the beginning of the prompt might lead to
responses that are incomplete, inflexible (giving the same answer even for different questions), incoherent or less accurate.

However, the outcome of where you position constraints may depend on the model and your specific use case.
You may get comparable responses irrespective of whether the constraints are positioned at the beginning or the end.

## Iterate and Repeat

The process of refining and improving prompts through successive iterations is crucial
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

## Reducing Hallucinations

Hallucinations in language models refer to the generation of incorrect or fictional information that is not present in the training data.
While achieving complete prevention is inherently difficult due to the dynamic and complex nature of natural language, there are strategies for effectively reducing hallucinations.

TODO

## Batching Prompts

Batching involves grouping multiple input prompts together and processing them simultaneously as a
batch.  This approach enhances efficiency by allowing the model to handle several prompts
at once to optimizing resource utilization.

In Lamini, the first argument to the runner can either be a single prompt string or a list of prompt strings.  When a single prompt is used, our system will run in non-batch mode and return a string output. On the other hand, when a prompt list used, the system will run in batch mode and return a list of dictionaries as output, like below.

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

Output:
```
[{'input': 'Is pizza nutritous?',
  'output': 'No, pizza is not typically considered a nutritious food due to its high calorie, carbohydrate, and fat content. However, it can be made healthier by using whole grain crust, lean protein toppings, and plenty of vegetables.'},
 {'input': 'Did Richard Nixon reisgn?',
  'output': ' Yes, Richard Nixon resigned as President of the United States on August 9, 1974.'},
 {'input: 'Summarize the impact of global warming.',
  'output': " Global warming has significant impacts on the Earth's environment, including rising sea levels, more frequent and intense heatwaves, droughts, and extreme weather events. It also affects wildlife, agriculture, and human health. The main cause of global warming is the increase in greenhouse gases in the atmosphere, primarily from human activities such as burning fossil fuels and deforestation. Addressing global warming requires reducing greenhouse gas emissions and transitioning to renewable energy sources."}]
```

## JSON Output with Lamini

While you can ask a model to output a json in the prompt, the results may lack consistency and reliability.
Hence, we've introduced a feature to guarantee valid JSON output through our web API.
See our [docs](https://lamini-ai.github.io/rest_api/completions)!

If you are interested in the technical details, see our blog post
[Guarantee Valid JSON Output with Lamini](https://www.lamini.ai/blog/guarantee-valid-json-output-with-lamini).