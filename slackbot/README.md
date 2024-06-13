<div align="center">
<img src="https://avatars.githubusercontent.com/u/130713213?s=200&v=4" width="110"><img src="https://huggingface.co/lamini/instruct-peft-tuned-12b/resolve/main/Lamini_logo.png?max-height=110" height="110">
</div>

# Slackbot

It is straightforward to call a LLM from Slack using Lamini.

First, get `<YOUR-LAMINI-API-KEY>` at [https://app.lamini.ai/account](https://app.lamini.ai/account).

Add the key to a config file (`slack/config.json`):

```json
{
    "api_endpoint": "https://api.lamini.ai",
    "SLACK_BOT_TOKEN": "xoxb-<YOUR SLACK BOT TOKEN>",
    "SLACK_APP_TOKEN": "xapp-<YOUR SLACK APP TOKEN>",
    "channel_token_mappings": {
        "<YOUR-SLACK-CHANNEL-ID>": {
            "_channel_name": "#channel-name",
            "token": "<YOUR-LAMINI-API-KEY>",
            "model_names": ["mistralai/Mistral-7B-Instruct-v0.2"]
        }
    }
}
```

# Start the slack bot

Start the slack bot.

```bash
./slackbot/bot-up.sh
```

Now mention your slackbot in a conversation.

# Edit the prompt

Edit the prompt in the [app](slack/app.py#L209C1-L220C1)

---

</div>
<div align="center">

![GitHub forks](https://img.shields.io/github/forks/lamini-ai/lamini-sdk) &ensp; © Lamini. &ensp; ![GitHub stars](https://img.shields.io/github/stars/lamini-ai/lamini-sdk)

</div>

--------
