# Instruction Fine Tuning (IFT)

## Introduction

Imagine you have a robot that talks about bears 🐻 but sometimes gets it a bit wrong. Fine-tuning is like showing the robot how to talk about bears in a way you like.

Consider this user prompt:

```
Tell me about a bear.
```

The robot replies:
```
Bears eat honey.
```

But what if we want the response to be more interesting?  We can train the model to do so with example instruction and output pairs.
For example, the `instruction` below guides the robot that when talking about a bear, describe it in an exciting manner, like a wild adventure.
The `desired output` serves as a guiding example for the model. It helps the model understand the preferred style, context, or content that you aim for in responses to specific prompts. However, it doesn't guarantee an exact replication of the desired output during actual usage.
```
[{"instruction": "Talk about a bear, but make it exciting like a wild adventure.",
  "desired output": Imagine Bigfoot, the fearless bear, roaming the vast forests, exploring caves, and conquering the great wilderness!"}
]
```

Now that we have trained the model, let's try this prompt again:
```
Tell me about a bear.
```

This time, the model produces an interesting response! 🤩

```
Meet Thunderpaws, the daring bear who explored untamed forests, facing roaring rivers and scaling towering mountains!
```

In this tutorial, we will teach you instruction fine tuning using
[generate_data.py](https://github.com/lamini-ai/sdk/blob/main/ift/generate_data.py),
a short program
that uses Lamini to load our fictional company's recent investment data,
chunks the data, and then
generate a list of [question, answer] pairs about the investments for training.
Each answer provides a guiding example for the corresponding question, similar to
the [instruction, desired output] pairs in the example above.

## Step 1: Load and Chunk Input Data

We start by loading the investment files and segmenting the text into chunks, where each
chunk will be used to create a new prompt.
This step is necessary because the model has limitations on the amount of text it
can process in a prompt.

The code below initializes a loader that specifies the data path
and how the data will be broken into chunks.
The first argument is the path to the data directory where the files will
be recursively loaded as text.  We use [../rag/data](https://github.com/lamini-ai/sdk/tree/main/rag/data),
the fictional investment data in the RAG tutorial.
In addition, we optionally specify `batch_size`, `chunk_size`, and `step_size`.
Please refer to [our RAG documentation] (add url to data to chunks section) for details on these  parameters.

```python
loader = DirectoryLoader(
    "../rag/data", # path to data directory
    batch_size=512,
    chunker=DefaultChunker(chunk_size=512, step_size=512),
)
```

Next, we iterate through the loader to generate and store the chunks.
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

In this example, we get 7 chunks.  Here is one of the chunks:
```
t only to their own growth but also to the broader discourse on responsible and ethical AI deployment.

On a parallel trajectory, Erlich B. played a pivotal role in guiding the seed round investment for
SeeFood, also situated in Palo Alto. The transaction concluded on October 1, 2023, with a substantial
investment totaling $10,000,000 and a 25% equity share. SeeFood stands out for its innovative use of
AI, creating engaging octopus cooking videos that can be experienced seamlessly through Oculus headsets. S
```

## Step 2: Generate Questions

In this step, we use `MistralRunner` to generate three
questions based on the investment data,
which uses the Mistral Instruct model.

```
runner = MistralRunner()
```

Next, we declare `Questions`, a class of Lamini `Type` with
three string fields: `question_1`, `question_2`, and `question_3`.
The string inside each `Context()` is an optional description of the
field.

```python
class Questions(Type):
    question_1: str = Context("")
    question_2: str = Context("")
    question_3: str = Context("")
```


For simplicity, we will use the chunk at index 4 only to demonstrate question generation.
```python
chunks = chunks[4:5] # range from index 4 to 5, but exclude item at index 5
```

The code below iterates through the chunks (we only have one chunk now).
For each chunk, it generates a prompt that is the concatenation the strings below:
1. The chunk wrapped in single quotes.
2. Newline (`\n`) to separate between the chunk and the next part.
3. An instruction to generate three diverse questions about the investments made by BigMoney Ventures based solely on the preceding single-quoted text.

We then execute `runner(...)` to generate `result` of type `Questions`
based on the prompt above and the specified system prompt, where each of
`result.question_1`, `result.question_2`, and `result.question_3` will
be a generated question.
At the end, we print out
the `questions` and append each [chunk, question] pair to `questions`.

```python
questions = []

for chunk in chunks:
    print(chunk)
    prompt = (
        "'"
        + chunk
        + "'\nThe preceeding single-quoted text is an excerpt describing various investments made by BigMoney Ventures. Generate three diverse questions about the investments.  Only generate questions that can be answered using information from the preceeding single-quoted text.  Do not ask questions that require additional information outside of the preceeding single-quoted text."
    )
    system_prompt = "You are an expert investment analyst working at BigMoney Ventures."

    result = runner(prompt, output_type=Questions, system_prompt=system_prompt)
    print("1.", result.question_1)
    print("2.", result.question_2)
    print("3.", result.question_3)
    questions.append([chunk, result.question_1])
    questions.append([chunk, result.question_2])
    questions.append([chunk, result.question_3])
```

Using the chunk containing investment data, the code passes the new prompt below to `runner`.
```
't only to their own growth but also to the broader discourse on responsible and ethical AI deployment.

On a parallel trajectory, Erlich B. played a pivotal role in guiding the seed round investment for
SeeFood, also situated in Palo Alto. The transaction concluded on October 1, 2023, with a substantial
investment totaling $10,000,000 and a 25% equity share. SeeFood stands out for its innovative use of AI,
creating engaging octopus cooking videos that can be experienced seamlessly through Oculus headsets. S'
The preceding single-quoted text is an excerpt describing various investments made by BigMoney Ventures.
Generate three diverse questions about the investments.  Only generate questions that can be answered
using information from the preceding single-quoted text.  Do not ask questions that require additional
information outside of the preceding single-quoted text.
```

The result is a `Questions` object with three fields:
```
question_1: What was the total investment amount for SeeFood and what was the equity share received by the investors 
question_2: What is the innovative use of AI that SeeFood is utilizing in their octopus cooking videos?`
question_3: What is the name of the company that Erlich B. played a pivotal role in guiding the seed round investment for SeeFood
```

## Step 3: Generate Answers

In this step, we generate answers for `questions`, which is the list of [chunk, question] pairs from the previosus step.

For each question, we create a new prompt by concatenating the following:
1. `question[0]` (the chunk) wrapped in single quotes.
2. New line `\n` to separate between the chunk in quotes and the next part.
3. A instruction that first describes the preceding text in single quotes as investment data from BigMoney Ventures, then directs the model to either answer the question or respond with "I don't know".
4. New lines `\n\n` to separate between the above from the next section.
5. `question[1]`, or a question generated from the previous step.


Next, similarly to how questions are generated, we proceed with executing `runner`
to generate `answer` based on the new prompt and the specified system prompt.
Lastly, we add the triple containing the chunk, question, and answer to `final_data_array`.  This array holds the necessary data for model training.

```python
final_data_array = []
# Generate Answers for each Answer
for index, question in enumerate(questions):
    # Run the model
    prompt = (
        "'"
        + question[0]  # chunk
        + "'\nThe preceding single-quoted text is an excerpt describing various investment made by BigMoney Ventures.  Answer the following question using information from the single-quoted text.  If you cannot answer the question using only the single-quoted text, respond only with the statement: \"I don't know.\"\n\n"
        + question[1]  # question about the chunk
    )
    system_prompt = "You are an expert in the field of investments."
    answer = runner(prompt, system_prompt=system_prompt)
    print(f"---------------------Answer for question {index}---------------------")
    print(answer)
    print(f"---------------------------------------------------------------------")
    final_data_array.append([question[0], question[1], answer])
```

Below is the chunk and the list of [question, answer] pairs we generated.
```
t only to their own growth but also to the broader discourse on responsible and ethical AI deployment.

On a parallel trajectory, Erlich B. played a pivotal role in guiding the seed round investment for
SeeFood, also situated in Palo Alto. The transaction concluded on October 1, 2023, with a substantial
investment totaling $10,000,000 and a 25% equity share. SeeFood stands out for its innovative use of
AI, creating engaging octopus cooking videos that can be experienced seamlessly through Oculus headsets. S
```

```
Question 1: What was the total investment amount for SeeFood and what was the equity share received by the investors
Answer: The total investment amount for SeeFood was $10,000,000 and the equity share received by the investors was 25%.

Question 2. What is the innovative use of AI that SeeFood is utilizing in their octopus cooking videos?
Answer: SeeFood is utilizing AI to create engaging octopus cooking videos that can be experienced seamlessly through Oculus headsets.

Question 3. What is the name of the company that Erlich B. played a pivotal role in guiding the seed round investment for SeeFood
Answer: The name of the company that Erlich B. played a pivotal role in guiding the seed round investment for SeeFood is SeeFood.
```

## Step 4: Save the Questions, Answers, and Data

We save the results in different formats in `qa_data/`.

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

## Try Out Instruction Fine Tuning

Experiment with
`python3 generate_data.py` to generate the output above.

Explore by trying different prompts, adjusting chunk parameters, or modifying the data directory! 🚀
