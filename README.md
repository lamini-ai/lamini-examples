# Lamini SDK

In this SDK, we include tutorials for achieving high-quality results with Language Models (LLMs) using Lamini.  With Lamini, you own the LLM you create -- you can deploy it or release it open source.  This SDK teaches effective tools for building LLMs.  We strongly encourage following the SDK *in order* as the concepts build on each other and are sorted by difficulty.

1. [Playground](https://github.com/lamini-ai/sdk/blob/main/01_playground/playground.md) - learn how to create a chat app that calls an LLM.  We include examples using slack, python, and react.
2. [Prompt Engineering](https://github.com/lamini-ai/sdk/blob/main/02_prompt_engineering/prompt_engineering.md) - crafting and refining input queries or instructions to achieve desired responses from language models.
3. [Retrieval Augmented Generation (RAG)](https://github.com/lamini-ai/sdk/blob/main/03_RAG/rag.md) - combining information retrieval with text generation to improve language models.
4. [Instruction Fine-Tuning (IFT)](https://github.com/lamini-ai/sdk/blob/main/04_IFT/ift.md) - refining language models through targeted adjustments to improve performance on specific tasks.

## Installation Instructions

Before you start, please get your Lamini API key and install the python library.

First, get `<YOUR-LAMINI-API-KEY>` at https://app.lamini.ai/account.
Add the key as an environment variable.
```
export LAMINI_API_KEY="<YOUR-LAMINI-API-KEY>"
```

Next, install the Python library.
```
pip install --upgrade lamini
```
