<div align="center">
<img src="https://avatars.githubusercontent.com/u/130713213?s=200&v=4" width="110"><img src="https://huggingface.co/lamini/instruct-peft-tuned-12b/resolve/main/Lamini_logo.png?max-height=110" height="110">
</div>

# Python Client

It is straightforward to call a LLM from python using Lamini.

First, get `<YOUR-LAMINI-API-KEY>` at [https://app.lamini.ai/account](https://app.lamini.ai/account).

Add the key as an environment variable. Or, authenticate via the Python library below.

```bash
export LAMINI_API_KEY="<YOUR-LAMINI-API-KEY>"
```

Install the Python library.

```python
pip install lamini
```

Run an LLM with a few lines of code.

```python
import lamini

llm_runner = lamini.LlamaV2Runner()
print(llm_runner.call("How are you?"))
```

<details>
<summary>Expected Output</summary>

"Hello! I'm just an AI, I don't have feelings or emotions like humans do, but I'm here to help you with any questions or concerns you may have. I'm programmed to provide respectful, safe, and accurate responses, and I will always do my best to help you. Please feel free to ask me anything, and I will do my best to assist you. Is there something specific you would like to know or discuss?"

</details>

Add the key to a config file ~/.lamini/configure.yaml

```yaml
production:
  key: <YOUR-LAMINI-API-KEY>
```

Instead of the environment variable, you can also pass your key in Python:

```python
lamini.api_key = "<YOUR-LAMINI-API-KEY>"
```

Run the Python Client example.
```
./answer_sample_questions.sh 
```

That's it! 🎉

---

</div>
<div align="center">

![GitHub forks](https://img.shields.io/github/forks/lamini-ai/lamini-sdk) &ensp; © Lamini. &ensp; ![GitHub stars](https://img.shields.io/github/stars/lamini-ai/lamini-sdk) 

</div>

--------