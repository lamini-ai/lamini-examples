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
https://github.com/lamini-ai/sdk/blob/ez-merge-earnings/05_data_pipeline/generate_data.py#L35

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

# Pipelines for performance

The Lamini LLM pipeline will automatically distribute your LLM calls over the entire cluster so you don't have
to think about thread pools and batching.

LLMs are extremely computationally intensive. Processing even a modest amount of data (e.g. GBs)
may require hundreds of GPUs to process quickly. So we recommend using this interface for any
data processing with more than ~100 LLM calls.

# Building Lamini pipeline

Use [the following code](generate_data.py#L40) to define a pipeline:

https://github.com/lamini-ai/lamini-examples/blob/fd355126bb73f7167098a2f7ba15488e23b7f945/05_data_pipeline/generate_data.py#L40-L50

It has two stages, QuestionGenerator, and AnswerGenerator.

## QuestionGenerator

The first stage reads a passage from an earnings call, and asks three questions about it.
Note how the code uses the output_type of the LLM to force it to generate three questions and automatically parse them.

https://github.com/lamini-ai/lamini-examples/blob/fd355126bb73f7167098a2f7ba15488e23b7f945/05_data_pipeline/generate_data.py#L53-L77

The question generator can be controlled by editing it's prompt.

https://github.com/lamini-ai/lamini-examples/blob/fd355126bb73f7167098a2f7ba15488e23b7f945/05_data_pipeline/generate_data.py#L113-L135

## AnswerGenerator

The answer generator is similar, just with a different prompt.  You can control it by editing the prompt.

https://github.com/lamini-ai/lamini-examples/blob/fd355126bb73f7167098a2f7ba15488e23b7f945/05_data_pipeline/generate_data.py#L192-L215
