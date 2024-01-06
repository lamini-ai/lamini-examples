from lamini import Lamini, LaminiIndex
import gzip
import logging
import json
import re
import requests
import time

import boto3
from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler

# Initializes your app with your bot token and socket mode handler
with open("./config.json", "r") as f:
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

    for index, model in enumerate(model_names):
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

    system_prompt = """\
You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."""

    headers = {
        "Authorization": "Bearer " + token,
        "Content-Type": "application/json",
    }

    body = {
        "id": "LaminiSlackbot",
        "model_name": model,
        "in_value": {
            "question": question,
            "system": system_prompt,
            },
        "out_type": {
            "Answer": "string",
        },
        "prompt_template": f"<s>[INST] <<SYS>>\n{input:system}\n<</SYS>>\n\n{input:question} [/INST]",
    }

    response = requests.post(
        config["api_endpoint"] + "/v1/completions",
        headers=headers,
        json=body,
    )

    if response.status_code == 200:
        answer = response.json()["Answer"]
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
