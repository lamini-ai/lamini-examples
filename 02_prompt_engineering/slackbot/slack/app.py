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

logging.basicConfig(filename="/tmp/docker.log", encoding="utf-8", level=logging.DEBUG)


@app.event("app_mention")
def main_event(client, event, say):
    channel_id = event["channel"]
    thread_ts = event.get("thread_ts", None) or event["ts"]
    question = re.sub("<[^>]+>", "", event["text"])  # Remove the @mention tag

    print("Mentioned in channel " + channel_id + " with question " + question)

    try:
        model_names = config["channel_token_mappings"][channel_id]["model_names"]
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

    task_description = "You are a medical coding expert with 20 years of experience at Mayo Clininc who has read the ICD11 standard from WHO. You are asked to code the following medical report. If you don't know the answer, reply that you don't know.  Answer in three sentences or less."

    headers = {
        "Authorization": "Bearer " + token,
        "Content-Type": "application/json",
    }

    task_parameters = f"Healthcare provider: {get_healthcare_provider()}.\nPatient Name: {get_patient_name()}.\nYear: {get_year()}."

    prompt = f"<s>[INST] {task_description}\n\n{task_parameters}\n\nAnswer the following question.\n\n{question} [/INST]"

    prompt = add_training_examples(prompt, task_description)

    print("Prompt:", prompt)

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
        answer = response.json()["output"]
        return answer
    else:
        print(response.status_code, response.reason)
        return f"Sorry I can't answer that: {response.status_code} {response.reason}"

def get_healthcare_provider():
    return "Mayo Clinic Rochester"

def get_patient_name():
    return "Sarah Williams"

def get_year():
    return "2023"


training_examples = [
    {
        "task_parameters": "Healthcare provider: Mayo Clinic Rochester.\nPatient Name: Sarah Williams.\nYear: 2023.",
        "question": "What is the diagnosis for alcohol induced anxiety disorder?",
        "answer": "6C40.71",
        "context": "Alcohol induced anxiety disorder is a type of anxiety disorder that is caused by or related to alcohol use. It is characterized by symptoms such as excessive worry, fear, or panic attacks that are triggered by the use of alcohol. These symptoms can interfere with the person's daily life and may lead to avoidance of social situations or other activities that involve alcohol.\n\nAlcohol induced anxiety disorder is a relatively rare condition, but it can be serious if left untreated. It is important for individuals who are experiencing symptoms of this disorder to seek professional help from a healthcare provider, such as a psychiatrist or therapist, who can evaluate their symptoms and provide appropriate treatment.\n\nTreatment for alcohol induced anxiety disorder typically involves a combination of therapy and medication. Cognitive-behavioral therapy (CBT) can be effective in helping individuals learn coping strategies and change negative thought patterns that contribute to their anxiety. Medications such as selective serotonin reuptake inhibitors (SSRIs) and benzodiazepines may also be prescribed to help manage symptoms.\n\nIt is important to note that alcohol induced anxiety disorder is just one type of anxiety disorder and that there are many other types of anxiety disorders that may be diagnosed based on a person's symptoms. It is always best to consult with a healthcare provider to determine the most appropriate diagnosis and treatment plan."
    },
    {
        "task_parameters": "Healthcare provider: Mayo Clinic Rochester.\nPatient Name: John Williams.\nYear: 2023.",
        "question": "How would you code swelling of the liver?",
        "answer": "ME10.0",
        "context": "Swelling of the liver, also known as hepatomegaly, is a condition in which the liver becomes enlarged. This can be caused by a variety of factors, including infections, inflammation, liver disease, and certain medications. Symptoms of hepatomegaly may include abdominal pain, nausea, vomiting, and jaundice (a yellowing of the skin and eyes).\n\nIt is important to note that swelling of the liver can be a serious condition and should be evaluated by a healthcare provider. A healthcare provider may perform a physical examination, blood tests, and imaging studies to determine the cause of the hepatomegaly and develop an appropriate treatment plan."
    },
    {
        "task_parameters": "Healthcare provider: Kaiser Permanente.\nPatient Name: Nancy Jones.\nYear: 2023.",
        "question": "Diabetes due to a genetic defect in insulin secretion is coded as what?",
        "answer": "5A13.1",
        "context": "Diabetes due to a genetic defect in insulin secretion is a type of diabetes that is caused by a mutation in the genes that control the production of insulin. Insulin is a hormone that is produced by the pancreas and is responsible for regulating blood sugar levels. In people with diabetes due to a genetic defect in insulin secretion, the pancreas does not produce enough insulin or the insulin that is produced is not effective at regulating blood sugar levels.\n\nSymptoms of diabetes due to a genetic defect in insulin secretion may include frequent urination, increased thirst, fatigue, and blurry vision. It is important to note that this type of diabetes is typically diagnosed in childhood or adolescence and is often associated with other genetic disorders.\n\nTreatment for diabetes due to a genetic defect in insulin secretion typically involves a combination of insulin therapy, dietary changes, and physical activity. In some cases, other medications or surgical procedures may also be used to manage the condition."
    },
]


def add_training_examples(prompt, system_prompt):
    complete_prompt = ""
    for example in training_examples:
        complete_prompt += f"<s>[INST] {system_prompt}\n\n{example['task_parameters']}\n\nAnswer the following question.\n\n{example['question']} [/INST]\n\n{example['answer']}\n\n{example['context']} </s>\n"
    return complete_prompt + prompt


def post_process(answer):
    clean_answer = answer.lstrip(" ")
    return clean_answer


# Start your app
if __name__ == "__main__":
    SLACK_APP_TOKEN = config["SLACK_APP_TOKEN"]
    SocketModeHandler(app, SLACK_APP_TOKEN).start()
