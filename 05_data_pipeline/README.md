<div align="center">
<img src="https://avatars.githubusercontent.com/u/130713213?s=200&v=4" width="110"><img src="https://huggingface.co/lamini/instruct-peft-tuned-12b/resolve/main/Lamini_logo.png?max-height=110" height="110">
</div>

# Data Pipelines

We can leverage Llama 3 to build data pipelines. Llama 3 can read english and
reason over it. We can use this capability to build data pipelines by inserting
calls to the LLM to perform data transformations.

Run the follow script to have Llama 3 read through earnings calls, pretend to
be a financial analyst, and ask relevant questions, and answer them using the
source text.

```bash
./scripts/generate-data.sh
```

We are only generating QA for the first line for this example since the transcript is massive.
Below is a sample of the output of the data pipeline.
The source code is in [generate_data.py](generate_data.py), and we'll walk through the code in the rest of this guide.

```json
{
  "company": "WPP",
  "question": "What is the percentage growth rate of WPP's business in Germany in Q1, according to Mark Read?",
  "answer": "16%"
}
{
  "company": "GDOT",
  "question": "What is the size of the asset size that GDOT aims to maintain to protect its revenue",
  "answer": "According to the transcript, GDOT aims to maintain an asset size of $10 billion or less to protect its revenue"
}

```

# Pipelines for performance and fault-tolerance

The Lamini LLM pipeline will automatically distribute your LLM calls over the entire cluster so you don't have
to think about thread pools and batching.

LLMs are extremely computationally intensive. Processing even a modest amount of data (e.g. GBs)
may require hundreds of GPUs to process quickly. So we recommend using this interface for any
data processing with more than ~100 LLM calls.

Pipeline also has automated retry to make sure transient failures do not break down the whole pipeline.

# Building Lamini pipeline

## Overview

A Lamini LLM pipeline is a series of stages.
Each stage is implemented as a subclass of `GenerationNode` class.
Each stage accepts an `AsyncGenerator` and produces another `AsyncGenerator`.

In this guide, the pipeline is defined in `QuestionAnswerPipeline`.
It has two stages: `QuestionGenerator` and `AnswerGenerator`, as shown in the `forward()` function below.

https://github.com/lamini-ai/lamini-examples/blob/70accea931ce666e3d1ca0b1609a745f085a7b70/05_data_pipeline/generate_data.py#L19-L33

We need to provide input to and save the results from the pipeline.
This is shown in `run_pipeline()` below, where the input was provided by `load_earnings_call()`,
and the results are saved by `save_answers()`:

https://github.com/lamini-ai/lamini-examples/blob/70accea931ce666e3d1ca0b1609a745f085a7b70/05_data_pipeline/generate_data.py#L148-L151

## Data Loading

The input to the pipeline is provided by `load_earnings_call()`, which is an `AsyncGenerator`,
because `GenerationNode` subclasses requires an input as `AsyncGenerator`.

https://github.com/lamini-ai/lamini-examples/blob/70accea931ce666e3d1ca0b1609a745f085a7b70/05_data_pipeline/generate_data.py#L121-L127

## QuestionGenerator

The first stage reads a passage from an earnings call, and ask LLMs to generate three questions about it.
This is achieved by [the prompt on line 79](https://github.com/lamini-ai/lamini-examples/blob/70accea931ce666e3d1ca0b1609a745f085a7b70/05_data_pipeline/generate_data.py#L79) in `make_prompt()` and `output_type` in `postprocess()`
to force it to generate three questions and automatically parse them:

https://github.com/lamini-ai/lamini-examples/blob/70accea931ce666e3d1ca0b1609a745f085a7b70/05_data_pipeline/generate_data.py#L52-L83

### preprocess & postprocess

One can define their own `preprocess()` to transform an `PromptObject` of a `GenerationNode` before passing it
to remote LLM inference API. Additionally, `postprocess()` to transfrom the result from LLM inference API.

In this example, `QuestionGenerator` has its own `preprocess()` & `postprocess()`:

https://github.com/lamini-ai/lamini-examples/blob/70accea931ce666e3d1ca0b1609a745f085a7b70/05_data_pipeline/generate_data.py#L48-L61

## AnswerGenerator

The answer generator is similar, just with a different prompt.  You can control it by editing the prompt.

https://github.com/lamini-ai/lamini-examples/blob/70accea931ce666e3d1ca0b1609a745f085a7b70/05_data_pipeline/generate_data.py#L85-L118

## Saving results

The output of the final `GenerationNode` is an `AsyncGenerator` that should be saved somewhere.
This is done in `save_answers()`, which uses `async for` to iterator through the results,
and write them into an output file.

https://github.com/lamini-ai/lamini-examples/blob/70accea931ce666e3d1ca0b1609a745f085a7b70/05_data_pipeline/generate_data.py#L129-L145
