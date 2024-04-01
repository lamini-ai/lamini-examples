<div align="center">
<img src="https://avatars.githubusercontent.com/u/130713213?s=200&v=4" width="110"><img src="https://huggingface.co/lamini/instruct-peft-tuned-12b/resolve/main/Lamini_logo.png?max-height=110" height="110">
</div>

# Guarantee Valid JSON Output with Lamini

Structured outputs are required to combine LLM outputs with code, e.g. to call a function with specific parameters.

In the past, engineers had to write custom parsers, which were complex, unreliable, and required re-engineering for each new model.

```code
from lamini import Lamini

llm = Lamini(model_name="meta-llama/Llama-2-7b-chat-hf")
llm.generate(
    "How old are you?",
    output_type={"age": "int", "units": "str"}
)
```


[1] JSONformer: https://github.com/1rgs/jsonformer

‍[2] OpenAI JSON Mode: https://platform.openai.com/docs/guides/text-generation/json-mode

---

</div>
<div align="center">

![GitHub forks](https://img.shields.io/github/forks/lamini-ai/lamini-sdk) &ensp; © Lamini. &ensp; ![GitHub stars](https://img.shields.io/github/stars/lamini-ai/lamini-sdk) 

</div>

--------