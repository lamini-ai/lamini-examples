## Simple Guide to Prompt Engineering

### Introduction

Prompt engineering involves designing prompts that elicit desired response
from the model. You can create prompts for a wide range of purposes, including
information retrieval, problem solving, creative/technical writing,
coding assistance, language translation, opinion generation,
text summerization, conversational agents, data analysis, and more.

For example, You can write a prompt like
"What was the decision in Nixon v. United States? Answer in one sentence."

Using lamini, you can find out the answer with a few lines of code.

```python
from llama import MistralRunner

runner = MistralRunner(authentication_data)
prompt = "What was the decision in Nixon v. United States? Answer in one sentence."
print(runner(prompt))
```

Sample output:

```
The decision in Nixon v. United States was that President Richard Nixon was ordered to release tapes of his conversations with his advisors, as part of a court order to comply with the Presidential Records Act.
```

