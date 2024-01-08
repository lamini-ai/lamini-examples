# React Playground

It is straightforward to call a LLM from a web app using Lamini.

First, get `<YOUR-LAMINI-API-KEY>` at [https://app.lamini.ai/account](https://app.lamini.ai/account).

Add the key to a config file ~/.playground/playground_config.yaml

```yaml
key: <YOUR-LAMINI-API-KEY>
url: https://api.powerml.co
```

# Start the playground

Start the playground.

```bash
cd 01_playground/react_llm_playground
./chat-playground-up
```

Navigate to the playground [http://localhost:5001](http://localhost:5001).

# Edit the prompt

Edit the prompt in the [python backend](playground/fastapi/chat_backend.py#L59C1-L60C1)

