# Slackbot

It is straightforward to call a LLM from Slack using Lamini.

First, get `<YOUR-LAMINI-API-KEY>` at [https://app.lamini.ai/account](https://app.lamini.ai/account).

Add the key to a config file:

```json
{
    "api_endpoint": "https://api.lamini.ai",
    "SLACK_BOT_TOKEN": "xoxb-<YOUR SLACK BOT TOKEN>",
    "SLACK_APP_TOKEN": "xapp-<YOUR SLACK APP TOKEN>",
    "channel_token_mappings": {
        "<YOUR-SLACK-CHANNEL-ID>": {
            "_channel_name": "#stream-greg",
            "token": "<YOUR-LAMINI-API-KEY>",
            "model_names": ["mistralai/Mistral-7B-Instruct-v0.1"]
        }
    }
}
```

# Start the slack bot

Start the slack bot.

```bash
cd 01_playground/slackbot
./bot-up.sh
```

Now mention your slackbot in a conversation.

# Edit the prompt

Edit the prompt in the [app](slack/app.py#L209C1-L220C1)


