# Instruction Fine Tuning (IFT)

## Introduction

Imagine you have a robot that talks about bears but sometimes gets it a bit wrong. Fine-tuning is like showing the robot how to talk about bears in a way you like.

Consider this user prompt:

```
Tell me about a bear.
```

The robot replies:
```
Bears eat honey.
```

But what if you want the response to be more interesting?  You can train the model with example instruction and output pairs.
For example, the `instruction` below guides the robot that when talking about a bear, describe it in an exciting manner, like a wild adventure.
The `desired output` serves as a guiding example for the model. It helps the model understand the preferred style, context, or content that you aim for in responses to specific prompts. However, it doesn't guarantee an exact replication of the desired output during actual usage.
```
[{"instruction": "Talk about a bear, but make it exciting like a wild adventure.",
  "desired output": Imagine Bigfoot, the fearless bear, roaming the vast forests, exploring caves, and conquering the great wilderness!"}
]
```

Let's try this prompt again:
```
Tell me about a bear.
```

This time, the model produces an interesting response!

```
"Meet Thunderpaws, the daring bear who explored untamed forests, facing roaring rivers and scaling towering mountains!"
```

In this tutorial, we guide you through program that uses Lamini to
read data, chunk the data,
and generate a list of [question, answer] pairs based on the data.
The final result can then be submitted to the model for trainig.

## Step 1: Load and Chunk Input Data

We start by loading the data files and segmenting the text into chunks,
which will be appended to prompts at a later stage.

The code below creates a loader that specifies the data path
and how the data will be broken into chunks.
The first argument is the path to the data directory where the files will
be recursively loaded as text.
The loader will fail if any file cannot be read as text.
In addition, we optionally specify `batch_size`, `chunk_size`, and `step_size`
in when initializing the loader.
Please refer to [our RAG documentation] (TODO: add link to data to chunks section) for details on these optional parameters.

```python
loader = DirectoryLoader(
    parent_directory + "/data",
    batch_size=512,
    chunker=DefaultChunker(chunk_size=512, step_size=512),
)
```

We then iterate through the loader to generate and store the chunks.
During each iteration, the loader yields a list of 512 (`batch_size`) chunks,
each with a length of 512 (`chunk_size`), and we concatenate this new list
to `chunks`.
We use [tqdm](https://github.com/tqdm/tqdm) when iterating over the loader,
which displays a progress bar for the iteration process.

```python
chunks = []
for chunk in tqdm(loader):
    chunks.extend(chunk)
```

## Step 2: Generate Questions

In this step, we will use the Mistral Instruct model to generate three
questions.

We specify that `MistralRunner` will be used to generate responses,
which uses the Mistral Instruct model.

```
runner = MistralRunner()
```

Next, we declare `Questions`, an object of Lamini `Type` with
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
For each chunk, it creates a new prompt with two sections:
1. The chunk wrapped in single quotes.
2. Newline (`'\n'`) followed by TODO.
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

## Step 3: Generate Answers

In this step, we generate answers for `questions`, the list of [chunk, question] pairs from the previous step.

For each question, we create a new prompt by concatenating the following:
1. `question[0]` (the chunk) wrapped in single quotes.
2. `\n` followed by TODO, followed by `\n\n`.
3. `question[1]`, or the question that corresponds to the chunk.

Next, similiarly to how questions are generated, we execute `runner`
to generate `answer` based on the new prompt and the specified system prompt.
Lastly, we add the triple containing the chunk, question, and answer to `final_data_array`.  This array holds the necessary data for model training.

```python
final_data_array = []
# Generate Answers for each Answer

for index, question in enumerate(questions):
    # Run the model
    prompt = (
        "'"
        + question[0]
        + "'\nThe preceeding single-quoted text is an excerpt from a MSA contract between Lamini and XXXXXXX.  Answer the following question using information from the single-quoted text.  If you cannot answer the question using only the single-quoted text, respond only with the statement: \"I don't know.\"\n\n"
        + question[1]
    )
    system_prompt = """You are a expert in the field of law."""
    answer = runner(prompt, system_prompt=system_prompt)

    print("---------------------- RUN PROMPT -------------------")
    print(prompt)
    print("---------------------- RUN PROMPT ------------------- DONE")    
    
    print(f"---------------------Answer for question {index}---------------------")
    print(answer)
    print(f"---------------------------------------------------------------------")
    final_data_array.append([question[0], question[1], answer])
```

## Step 4: Save the Questions, Answers, and Data

You can save the results from the previous step to a csv file or a json file,
as shown below.

```python
# Save the questions, answers, and data in a csv file (logging)
with open("qa_data/generated_data.csv", "w") as f:
    writer = csv.writer(f)
    writer.writerow(["Data", "Questions", "Answers"])
    for data in final_data_array:
        writer.writerow(data)

# And save to a json file for logging
with open("qa_data/generated_data.json", "w") as f:
    json.dump(final_data_array, f, indent=4)
```

Finally, save the format that will actually be submitted to the model for finetuning.

```python
training_data = [
    [
        {"data": data[0], "question": data[1]},
        {"answer": data[2]},
    ]
    for data in final_data_array
]

with open("qa_data/generated_data_finetuning.json", "w") as f:
    json.dump(training_data, f, indent=4)
```