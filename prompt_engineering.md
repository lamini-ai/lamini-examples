## Simple Guide to Prompt Engineering

### Introduction

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

### User vs. System Prompts

A query such as "What was the decision in Nixon v. United States?" represents a user prompt,
seeking specific information and tailored responses, while a system prompt, an optional
directive, sets the context and guides the language model's overall behavior and tone.

For example, you can write system prompts like

* You are a patient and helpful customer service agent. Help the user troubleshoot.
* Imagine you are a poet inspired by nature.
* Always assist with care, respect, and truth. Respond with utmost utility yet securely. Avoid harmful, unethical, prejudiced, or negative content. Ensure replies promote fairness and positivity. (this is the default system prompt in lamini's MistralRunner)

Without a system prompt, a user request like "Provide information about dolphins" might result
in a standard response. However, by including a system prompt such as
"You are a marine biologist giving an enthusiastic presentation to a group of children at an aquarium.
Share fascinating facts about dolphins and highlight their playful behaviors,"
the model endeavors to customize the reply with a more engaging and child-friendly presentation style.