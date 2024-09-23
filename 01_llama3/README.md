<div align="center">
<img src="https://avatars.githubusercontent.com/u/130713213?s=200&v=4" width="110"><img src="https://huggingface.co/lamini/instruct-peft-tuned-12b/resolve/main/Lamini_logo.png?max-height=110" height="110">
</div>

# Python Client

It is straightforward to call powerful LLMs like Llama3 from Python using Lamini.

1. Get `<YOUR-LAMINI-API-KEY>` at [your account page on Lamini](https://app.lamini.ai/account).
2. Modify `llama3.py` to insert the key.
   https://github.com/lamini-ai/lamini-sdk/blob/main/01_llama3/llama3.py#L1-L6

   Alternatively, you can fill in the key in the `~/.lamini/configure.yaml` as follows,
   persisting your key whenever you use Lamini:
   ```bash
   production:
     key: <YOUR-LAMINI-API-KEY>
   ```

That's it! 🎉
```bash
cd 01_llama3
python3 llama3.py
```
