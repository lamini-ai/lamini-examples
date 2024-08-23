<div align="center">
<img src="https://avatars.githubusercontent.com/u/130713213?s=200&v=4" width="110"><img src="https://huggingface.co/lamini/instruct-peft-tuned-12b/resolve/main/Lamini_logo.png?max-height=110" height="110">
</div>

# Slackbot

# To create your own bot
* Go to https://api.slack.com/apps?new_app=1
* Choose "From an app manifest"
* Give it a name and set `"socket_mode_enabled"` to be `true`
* Under "OAuth & Permissions", add the following scopes: `app_mentions:read`, `channels:history`, `chat:write`, `commands`, `groups:history`, `reactions:read`, `reactions:write`
* Under "Event Subscriptions", enable Events and subscribe to the following bot events: `app_mention`, `reaction_added`, `reaction_removed`
* (optional) If you want the count slash command to work, add `/count-reactions` under "Slash Commands"
* Go to `Basic Information`, install your app and add an App-Level token with all Scopes
* Add the bot to your channel using `/add` in Slack

# To run the bot
* Clone and cd into this repository
* Update `config.json` with your Lamini and Slack information (set up above)
    * To get the channel id, go to the channel in Slack and copy the last part of the channel link
    * First, get `<YOUR-LAMINI-API-KEY>` at [https://app.lamini.ai/account](https://app.lamini.ai/account).

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
            "model_names": ["mistralai/Mistral-7B-Instruct-v0.3"]
        }
    }
}
```

# Start the slack bot

Start the slack bot.

```bash
./slackbot/bot-up.sh
```

You can also run using Docker:
* `docker build --tag 'slack_bot' .`
* `docker run -d 'slack_bot'`

Now mention your slackbot in a conversation.

# Edit the prompt

Edit the prompt in the [app.py](https://github.com/lamini-ai/lamini-sdk/blob/main/12_slackbot/slack/app.py#L211-L214).

---

</div>
<div align="center">

![GitHub forks](https://img.shields.io/github/forks/lamini-ai/lamini-sdk) &ensp; © Lamini. &ensp; ![GitHub stars](https://img.shields.io/github/stars/lamini-ai/lamini-sdk)

</div>

--------
