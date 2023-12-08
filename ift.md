# Instruction Fine Tuning

## Introduction

Instruction Fine-Tuning is used to make language models better at following instructions by training them on examples where instructions are paired with desired outcomes.

Consider this prompt:
```
Generate a recipe suggestion for a user interested in trying a vegetarian dish.
```

Without fine tuning, the model may respond with a recipe
without considering the user's preferences or dietary restrictions. :thumbsdown:

Let's try fine tuning with some example pairs.

```
[{"instruction": "Provide a vegetarian recipe option for someone who enjoys spicy flavors, with a focus on quick preparation and minimal ingredients.",
 "desired_output": "Encourage the model to consider user preferences for spice level, preparation time, and simplicity."
 },
 {"instruction": "Suggest a vegetarian recipe suitable for a user following a gluten-free diet, emphasizing diverse textures and flavors without compromising dietary restrictions."
 "desired_output": "Guide the model to consider specific dietary needs and preferences."
 }
]
```

Improved Response (after fine-tuning): The model, having learned from these fine-tuned examples, generates vegetarian recipe suggestions that align with users' preferences, ensuring a more personalized and enjoyable cooking experience.

TODO: add more, mayb example

In this tutorial, we will show you how to use Lamini for IFT.
We will show a sample program that reads your data, chunks the data,
and generate a list of [question, answer] pairs based on the data.
You can then use the pairs for training.

## Step 1: Load and Chunk Input Data

We start by loading the data files and segmenting the text into chunks,
which will be appended to prompts at a later stage.
Dividing the text into chunks (ex. 512 characters each) is necessary because
appending a large text to prompts exceed the model's input limitations.

The code below creates a loader that specifies the data path
and how the data will be broken into chunks.
The first argument is the path to the data directory, where files will
be recursively loaded.
This directory should only contain files that can be loaded as text.
Otherwise, the loader will fail (TODO: double check).
In addition, we optionally specify `batch_size`, `chunk_size`, and `step_size`
in when initializing the loader.
Please refer to [TODO: add link] for details on these optional parameters.

TODO: find out what batch_size is here.

```python
loader = DirectoryLoader(
    parent_directory + "/data",
    batch_size=512,
    chunker=DefaultChunker(chunk_size=512, step_size=512),
)
```

We then iterate through the loader to generate and store the chunks.

```python
chunks = []
for chunk in tqdm(loader):
    chunks.extend(chunk)
```

During each iteration, the loader yields a list of 512 (`batch_size`) chunks,
each with a length of 512 (`chunk_size`), and we concatenate this new list
to `chunks`.
We use [tqdm](https://github.com/tqdm/tqdm] when iterating over the loader,
which displays a progress bar for iteration process.

## Step 2: Generate Questions

In this step, we will use the Mistral Instruct model to generate three
questions.

We specify that `MistralRunner` will be used to generate responses,
which uses the Mistral Instruct model.

```
runner = MistralRunner()
```

Next, we declare `Questions`, an object of lamini `Type` with
three string fields: `question_1`, `question_2`, and `question_3`.

```python
class Questions(Type):
    question_1: str = Context("")
    question_2: str = Context("")
    question_3: str = Context("")
```

For simplicity, we use the chunk at index 2 only to demonstrate the question generation
```python
chunks = chunks[2:3] # range from index 2 to 3, but exclude item at index 3
```

The code below iterates through the chunks (we only have one chunk now).
For each chunk, it creates a new prompt, where the first part is
the chunk wrapped in single quotes, the second part is newline `\n` followed
by TODO.
For example, TODO.
We then execute `runner(...)` to generate `result` of type `Questions`
based on the new prompt and the specific system prompt.

The code also prints out each question, which are TODO.
At the end, `questions ` contains a list of [chunk, question] pairs.
For example, TODO.

```python
questions = []
...
for chunk in chunks:
    prompt = (
        "'"
        + chunk
        + "'\nThe preceeding single-quoted text is an excerpt from a MSA contract between Lamini and XXXXX. Generate three diverse questions about the MSA.  Only generate questions that can be answered using information from the preceeding single-quoted text.  Do not ask questions that require additional information outside of the preceeding single-quoted text."
    )
    system_prompt = """You are an expert contract analyst working at Point32 health."""

    result = runner(prompt, output_type=Questions, system_prompt=system_prompt)
    print("1.", result.question_1)
    print("2.", result.question_2)
    print("3.", result.question_3)
    questions.append([chunk, result.question_1])
    questions.append([chunk, result.question_2])
    questions.append([chunk, result.question_3])
```