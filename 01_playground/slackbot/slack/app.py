from lamini import Lamini

import logging
import json
import re
import requests
import time

from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler

# Initializes your app with your bot token and socket mode handler
with open("/app/lamini-slackbot/slack/config.json", "r") as f:
    config = json.load(f)

SLACK_BOT_TOKEN = config["SLACK_BOT_TOKEN"]
app = App(token=SLACK_BOT_TOKEN)  # Store slack bot and app token

model_reaction_counts = (
    {}
)  # dict where key is the channel, value is a dict of arrays where the key corresponds to the model and the value is [# thumbs-up, # thumbs-down]

logging.basicConfig(filename="/tmp/docker.log", encoding='utf-8', level=logging.DEBUG)

@app.event("app_mention")
def main_event(client, event, say):
    channel_id = event["channel"]
    thread_ts = event.get("thread_ts", None) or event["ts"]
    question = re.sub("<[^>]+>", "", event["text"])  # Remove the @mention tag

    print("Mentioned in channel " + channel_id + " with question " + question)

    try:
        model_names = config["channel_token_mappings"][channel_id][
            "model_names"
        ]
    except:
        say(
            "Channel mapping does not exist or is incorrect: check config",
            thread_ts=thread_ts,
        )
        return

    print("Model names: " + str(model_names))

    for index, model in enumerate(model_names):
        try:
            loading = say("_Typing..._", thread_ts=thread_ts)

            answer = ask_model_question(channel_id, model, question)
            clean_answer = post_process(answer)

            if len(model_names) == 1:
                text = clean_answer
            else:
                text = f"*Model {index + 1}:*\n" + clean_answer

            print(clean_answer)
            reply = client.chat_update(
                channel=loading["channel"],
                ts=loading["ts"],
                text=text,
            )
            print(reply)
            client.reactions_add(
                name="thumbsup",
                channel=reply["channel"],
                timestamp=reply["ts"],
            )
            client.reactions_add(
                name="neutral_face",
                channel=reply["channel"],
                timestamp=reply["ts"],
            )
            client.reactions_add(
                name="thumbsdown",
                channel=reply["channel"],
                timestamp=reply["ts"],
            )
        except Exception as e:
            print(e)
            continue


@app.event("reaction_added")
def reaction_event(client, event):
    channel = event["item"]["channel"]
    try:
        message = client.conversations_replies(
            channel=channel,
            ts=event["item"]["ts"],
            inclusive=True,  # Limit the results to only one
            limit=1,
        )

        if "bot_id" not in message["messages"][0]:  # check that message was from bot
            return

        print("Got reaction " + event["reaction"] + " on ")
        print(message)

        try:
            model_number = re.search(
                "Model (.*):", message["messages"][0]["text"]
            ).group(1)
        except:
            model_number = -1

        if channel not in model_reaction_counts:
            model_reaction_counts[channel] = {}

        model_count = model_reaction_counts[channel]

        if "+1" in event["reaction"]:
            if model_number in model_count:
                model_count[model_number][0] = model_count[model_number][0] + 1
            else:
                model_count[model_number] = [1, 0, 0]
        elif "-1" in event["reaction"]:
            if model_number in model_count:
                model_count[model_number][2] = model_count[model_number][2] + 1
            else:
                model_count[model_number] = [0, 0, 1]
        elif event["reaction"] == "neutral_face":
            if model_number in model_count:
                model_count[model_number][1] = model_count[model_number][1] + 1
            else:
                model_count[model_number] = [0, 1, 0]
    except:
        message = "Couldn't find message"
        print(message)


@app.event("reaction_removed")
def reaction_remove_event(client, event):
    channel = event["item"]["channel"]
    try:
        message = client.conversations_replies(
            channel=channel,
            ts=event["item"]["ts"],
            inclusive=True,  # Limit the results to only one
            limit=1,
        )

        if "bot_id" not in message["messages"][0]:  # check that message was from bot
            return

        print("Removed reaction " + event["reaction"] + " on ")
        print(message)

        try:
            model_number = re.search(
                "Model (.*):", message["messages"][0]["text"]
            ).group(1)
        except:
            model_number = -1

        if channel not in model_reaction_counts:
            model_reaction_counts[channel] = {}

        model_count = model_reaction_counts[channel]

        if "+1" in event["reaction"]:
            if model_number in model_count:
                model_count[model_number][0] = model_count[model_number][0] - 1
            else:
                model_count[model_number] = [1, 0, 0]
        elif "-1" in event["reaction"]:
            if model_number in model_count:
                model_count[model_number][2] = model_count[model_number][2] - 1
            else:
                model_count[model_number] = [0, 0, 1]
        elif event["reaction"] == "neutral_face":
            if model_number in model_count:
                model_count[model_number][1] = model_count[model_number][1] - 1
            else:
                model_count[model_number] = [0, 1, 0]
    except:
        message = "Couldn't find message"
        print(message)


@app.command("/count-reactions")
def get_count_command(ack, body, respond):
    ack()
    channel = body["channel_id"]
    text = ""
    try:
        model_count = model_reaction_counts[channel]
        for key, val in dict(sorted(model_count.items())).items():
            if key == -1:
                text += (
                    f"🦙 Model currently has {val[0]} 👍, {val[1]} 😐, and {val[2]} 👎\n"
                )
            else:
                text += f"🦙 Model {key} currently has 👍: {val[0]}, 😐: {val[1]}, and 👎: {val[2]}\n"
    except:
        text = "Issue getting counts, have any reactions been added?"

    respond(
        {
            "text": text,
            "response_type": "in_channel",
        }
    )


def ask_model_question(channel_id, model, question):
    token = config["channel_token_mappings"][channel_id]["token"]

    system_prompt = "You are a helpful medical assistant."

    headers = {
        "Authorization": "Bearer " + token,
        "Content-Type": "application/json",
    }

    prompt = f"<s>[INST] {system_prompt}\n\nAnswer the following question.\n\n{question} [/INST]"

    body = {
        "id": "LaminiSDKSlackbot",
        "model_name": model,
        "prompt": prompt,
    }

    response = requests.post(
        config["api_endpoint"] + "/v1/completions",
        headers=headers,
        json=body,
    )

    if response.status_code == 200:
        answer = response.json()['output']
        return answer
    else:
        print(response.status_code, response.reason)
        return f"Sorry I can't answer that: {response.status_code} {response.reason}"


def post_process(answer):
    clean_answer = answer.lstrip(' ')
    return clean_answer


# Start your app
if __name__ == "__main__":
    SLACK_APP_TOKEN = config["SLACK_APP_TOKEN"]
    SocketModeHandler(app, SLACK_APP_TOKEN).start()
