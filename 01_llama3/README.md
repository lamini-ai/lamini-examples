# Python Client

It is straightforward to call powerful LLMs like Llama3 from Python using Lamini.

1. Get `<YOUR-LAMINI-API-KEY>` at [your account page on Lamini](https://app.lamini.ai/account).
2. Modify `llama3.py` to insert your key.

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
