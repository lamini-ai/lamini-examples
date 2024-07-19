<div align="center">
<img src="https://avatars.githubusercontent.com/u/130713213?s=200&v=4" width="110"><img src="https://huggingface.co/lamini/instruct-peft-tuned-12b/resolve/main/Lamini_logo.png?max-height=110" height="110">
</div>

# Python Client

It is straightforward to call powerful LLMs like Llama3 from Python using Lamini.

1. Get `<YOUR-LAMINI-API-KEY>` at [your account page on Lamini](https://app.lamini.ai/account).
2. Modify `llama3.py` to insert the key.
   https://github.com/lamini-ai/lamini-sdk/blob/main/01_llama3/llama3.py#L1-L6
3. Run using Docker:
    ```bash
    ./scripts/llama3.sh
    ```

That's it! 🎉 You can also install lamini SDK and run Python code directly:
```bash
pip install lamini
python llama3.py`
```
