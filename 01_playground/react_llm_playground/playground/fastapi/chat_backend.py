
from fastapi import FastAPI, Request, Body

from fastapi.middleware.cors import CORSMiddleware

from playground.fastapi.generate_payload import GeneratePayload
from playground.util.get_config import create_config, get_config

import copy
import requests
import json

import logging

logging.basicConfig(level=logging.INFO)

app = FastAPI()

origins = [
    "http://localhost:5000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/streaming_generate")
def streaming_generate(
    payload: GeneratePayload = Body(...),
):
    payload = edit_prompts(payload)

    result = run_streaming_inference(payload)

    return result

@app.post("/generate")
def generate(
    payload: GeneratePayload = Body(...),
):
    result = run_generate(payload, db)

    return result

@app.on_event("startup")
def startup():
    create_config()

def edit_prompts(payload):
    new_payload = copy.deepcopy(payload)

    new_payload.in_value = [edit_prompt(prompt) for prompt in payload.in_value]

    return new_payload

def edit_prompt(prompt_object):

    new_prompt = {
        "{input:task_description}": "You are a helpful assistant.\n\n",
    }

    prompt_object = copy.deepcopy(prompt_object)

    for key, value in new_prompt.items():
        prompt_object["question"] = prompt_object["question"].replace(key, value)

    return prompt_object

def run_streaming_inference(payload):
    config = get_config()
    json_payload = json.loads(payload.json())
    json_payload["id"] = "streaming_generate"
    json_payload["model_config"] = {}
    logging.info("json_payload to streaming completions: %s", json_payload)
    return make_web_request(
        key=config["key"],
        url=config["url"] + "/v1/streaming_completions",
        http_method="post",
        json=json_payload,
    )

class LlamaError(Exception):
    def __init__(
        self,
        message=None,
    ):
        super(LlamaError, self).__init__(message)


class ModelNameError(LlamaError):
    """The model name is invalid. Make sure it's a valid model in Huggingface or a finetuned model"""


class APIError(LlamaError):
    """There is an internal error in the Lamini API"""


class AuthenticationError(LlamaError):
    """The Lamini API key is invalid"""


class RateLimitError(LlamaError):
    """The QPS of requests to the API is too high"""


class UserError(LlamaError):
    """The user has made an invalid request"""


class UnavailableResourceError(LlamaError):
    """Model is still downloading"""


class ServerTimeoutError(LlamaError):
    """Model is still downloading"""

def make_web_request(key, url, http_method, json):
    headers = {
        "Content-Type": "application/json",
        "Authorization": "Bearer " + key,
    }
    if http_method == "post":
        resp = requests.post(url=url, headers=headers, json=json)
    elif http_method == "get":
        resp = requests.get(url=url, headers=headers)
    else:
        raise Exception("http_method must be 'post' or 'get'")
    try:
        resp.raise_for_status()
    except requests.exceptions.HTTPError as e:
        print("status code:", resp.status_code)
        if resp.status_code == 404:
            try:
                json_response = resp.json()
            except Exception:
                json_response = {}
            raise ModelNameError(json_response.get("detail", "ModelNameError"))
        if resp.status_code == 429:
            try:
                json_response = resp.json()
            except Exception:
                json_response = {}
            raise RateLimitError(json_response.get("detail", "RateLimitError"))
        if resp.status_code == 401:
            try:
                json_response = resp.json()
            except Exception:
                json_response = {}
            raise AuthenticationError(
                json_response.get("detail", "AuthenticationError")
            )
        if resp.status_code == 400:
            try:
                json_response = resp.json()
            except Exception:
                json_response = {}
            raise UserError(json_response.get("detail", "UserError"))
        if resp.status_code == 503:
            try:
                json_response = resp.json()
            except Exception:
                json_response = {}
            raise UnavailableResourceError(
                json_response.get("detail", "UnavailableResourceError")
            )
        if resp.status_code != 200:
            try:
                description = resp.json()
            except BaseException:
                description = resp.status_code
            finally:
                if description == {"detail": ""}:
                    raise APIError("500 Internal Server Error")
                raise APIError(f"API error {description}")

    return resp.json()
