<div align="center">
<img src="https://avatars.githubusercontent.com/u/130713213?s=200&v=4" width="110"><img src="https://huggingface.co/lamini/instruct-peft-tuned-12b/resolve/main/Lamini_logo.png?max-height=110" height="110">
</div>
<div align="center">

[![Latest Release](https://img.shields.io/badge/Latest%20Version-1.4.3-blue?logo=github)](https://github.com/lamini-ai/lamini-sdk/commits/main)
[![GitHub License](https://img.shields.io/github/license/lamini-ai/lamini)](https://github.com/lamini-ai/lamini-sdk/blob/main/LICENSE)</div>

## Lamini Examples

*Note that this repo is currently being re-worked. Please excuse our appearance as we improve our docs and examples!*

In this repo, we include tutorials for achieving high-quality results with Language Models (LLMs) using Lamini.</br>  With Lamini, you own the LLM you create -- you can deploy it or release it open source.</br>  These examples show effective tools for building LLMs.</br>  We strongly encourage following the examples *in order* as the concepts build on each other and are sorted by difficulty.

1. [Playground](01_playground/playground.md) - learn how to create a chat app that calls an LLM.  We include examples using slack, python, and react.
2. [Prompt Engineering](02_prompt_engineering/prompt_engineering.md) - crafting and refining input queries or instructions to achieve desired responses from language models.
3. [Retrieval Augmented Generation (RAG)](03_RAG/rag.md) - combining information retrieval with text generation to improve language models.
4. [Instruction Fine Tuning (IFT)](04_IFT/ift.md) - train your LLM using data in a question and answer format.
5. [JSON](05_json/json.md) - extract structured output from an LLM, following a guaranteed JSON schema.
6. [Evaluation](06_json/eval.md) - evaluate the quality of your LLM.
7. [Classify](07_classify/classify.md) - classify data using an LLM, e.g. to filter out low quality training data
8. [Domain Adaptation](#) - continue training your LLM using a large amount of new data. COMING SOON.
9. [Pretraining](#) - Train an LLM from scratch.  Make sure you have 1,000s of GPUs allocated. COMING SOON.

### Notes

The goal of this repo is to teach and provide examples of important tools for building LLMs; the examples emphasize simplicitly and readibility, not heavy optimization.</br>  Once you have mastered a module from this repo, consider forking it and adapting it to your own application.</br>  All of the code in this repository is licensed Apache 2. You are free to use it for any purpose including commercial applications.

### Installation Instructions

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

### GitHub Repository
---
The source code for this repo can be found on GitHub at [lamini-ai/lamini-sdk](https://github.com/lamini-ai/lamini-sdk). Feel free to explore and contribute!

### About Lamini
---
Lamini is the LLM platform for developers to specialize LLMs on their own data and infrastructure: easier, faster, and better than any LLM for their use case.</br> Our mission is to build customizable superintelligence that anyone can build and own.

---

</div>
<div align="center">

![GitHub forks](https://img.shields.io/github/forks/lamini-ai/lamini-sdk) &ensp; © Lamini &ensp; ![GitHub stars](https://img.shields.io/github/stars/lamini-ai/lamini-sdk)

</div>

--------
