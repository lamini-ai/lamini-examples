<div align="center">
<img src="https://avatars.githubusercontent.com/u/130713213?s=200&v=4" width="110"><img src="https://huggingface.co/lamini/instruct-peft-tuned-12b/resolve/main/Lamini_logo.png?max-height=110" height="110">
</div>

# Guarantee Valid JSON Output with Lamini

Structured outputs are required to combine LLM outputs with code, e.g. to call a function with specific parameters.

In the past, engineers had to write custom parsers, which were complex, unreliable, and required re-engineering for each new model. With Lamini, you can easily get guaranteed json output.

https://github.com/lamini-ai/sdk/blob/main/json_output/llm_json.py#L1-L9

# Run with structured output

```bash
./scripts/run_llm_json.sh
```

Try editing the `output_type` in `json_output/llm_json.py` and see what happens!

---

</div>
<div align="center">

![GitHub forks](https://img.shields.io/github/forks/lamini-ai/lamini-sdk) &ensp; © Lamini. &ensp; ![GitHub stars](https://img.shields.io/github/stars/lamini-ai/lamini-sdk)

</div>

--------
