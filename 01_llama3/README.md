<div align="center">
<img src="https://avatars.githubusercontent.com/u/130713213?s=200&v=4" width="110"><img src="https://huggingface.co/lamini/instruct-peft-tuned-12b/resolve/main/Lamini_logo.png?max-height=110" height="110">
</div>

# Python Client

It is straightforward to call powerful LLMs like Llama3 from python using Lamini.

First, get `<YOUR-LAMINI-API-KEY>` at [https://app.lamini.ai/account](https://app.lamini.ai/account).

Add the key as an environment variable. Or, authenticate via the Python library below.

Install the Python library.

```python
pip install lamini
```

Run Llama3 with a few lines of code.

```python
import lamini

lamini.api_key = "<YOUR-LAMINI-API-KEY>"

llm = lamini.Lamini(model_name="meta-llama/Meta-Llama-3-8B-Instruct")
print(llm.generate("How are you?"))
```

<details>
<summary>Expected Output</summary>

"Hello! I'm just an AI, I don't have feelings or emotions like humans do, but I'm here to help you with any questions or concerns you may have. I'm programmed to provide respectful, safe, and accurate responses, and I will always do my best to help you. Please feel free to ask me anything, and I will do my best to assist you. Is there something specific you would like to know or discuss?"

</details>


You can run this script to run the entire process setting up a clean environment in Docker.

```bash
./llama3.sh
```

That's it! 🎉

