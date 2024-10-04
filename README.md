<div align="center">
<img src="https://avatars.githubusercontent.com/u/130713213?s=200&v=4" width="110"><img src="https://huggingface.co/lamini/instruct-peft-tuned-12b/resolve/main/Lamini_logo.png?max-height=110" height="110">
</div>
<div align="center">

[![Latest Release](https://img.shields.io/badge/Latest%20Version-1.4.3-blue?logo=github)](https://github.com/lamini-ai/lamini-sdk/commits/main)
[![GitHub License](https://img.shields.io/github/license/lamini-ai/lamini)](https://github.com/lamini-ai/lamini-sdk/blob/main/LICENSE)</div>
## Installation Instructions

1. Before you start, please get your `<YOUR-LAMINI-API-KEY>` at https://app.lamini.ai/account.
Create `~/.lamini/configure.yaml` and write the file as follows, this will persist your Lamini credentials:
```yaml
production:
  key: <YOUR-LAMINI-API-KEY>
```

2. Install python requirements
```bash
pip3 install -r ./lamini-examples/requirements.txt
```

## Lamini Examples

In this repo, we include tutorials for achieving high-quality results with Language Models (LLMs), like Llama3, using Lamini. With Lamini, you own the LLM you create -- you can deploy it or release it open source.

These examples show effective tools for building LLMs. We strongly encourage following the examples *in order* as the concepts build on each other and are sorted by difficulty.

1. [Llama3](01_llama3/README.md) - Generate text with Llama3, a powerful LLM.
2. [Evaluation](02_eval/README.md) - Evaluate the quality of your LLM.
3. [Prompt Tuning](03_prompt_tuning/README.md) - Improve the quality of your LLM by tuning the prompts you use.
4. [RAG Tuning](04_rag_tuning/README.md) - Retrieval Augmented Generation (RAG), improve the quality of your LLM by tuning the retrieval component.
5. [Data Pipeline](05_data_pipeline/README.md) - Prepare your data for training an LLM.
6. [Memory Tuning](06_memory_tuning/README.md) - Memory tune an LLM on your data.

Extra Examples:
- [Meta Llama Memory Tuning 🔗](https://github.com/meta-llama/llama-recipes/tree/main/recipes/3p_integrations/lamini/text2sql_memory_tuning) - Embed facts into an LLM and improve factual accuracy on your data.
- [JSON Output](10_json_output/README.md) - Extract structured output from an LLM, following a guaranteed JSON schema.
- [Classify](11_classify/README.md) - Classify data using an LLM, e.g. to filter out low quality training data
- [Slackbot](12_slackbot/README.md) - Learn how to create a Slack bot that calls an LLM.

### Notes

The goal of this repo is to teach and provide examples of important tools for building LLMs; the examples emphasize simplicitly and readibility, not heavy optimization.</br>  Once you have mastered a module from this repo, consider forking it and adapting it to your own application.</br>  All of the code in this repository is licensed Apache 2. You are free to use it for any purpose including commercial applications.

### GitHub Repository
---
The source code for this repo can be found on GitHub at [lamini-ai/lamini-examples](https://github.com/lamini-ai/lamini-examples). Feel free to explore and contribute!

### About Lamini
---
Lamini is the LLM platform for developers to specialize LLMs on their own data and infrastructure: easier, faster, and better than any LLM for their use case.</br> Our mission is to build customizable superintelligence that anyone can build and own.

---

</div>
<div align="center">

![GitHub forks](https://img.shields.io/github/forks/lamini-ai/lamini-sdk) &ensp; © Lamini &ensp; ![GitHub stars](https://img.shields.io/github/stars/lamini-ai/lamini-sdk)

</div>

--------
