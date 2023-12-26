import requests
import random
import jsonlines
import json
import urllib3
from concurrent.futures import ThreadPoolExecutor

from tqdm import tqdm

urllib3.disable_warnings()


def main():
    max_entities = 1000
    max_parallel_requests = 64

    random.seed(42)

    token = get_access_token()

    visited_entities = set()
    unvisited_entities = set(["https://id.who.int/icd/release/11/2023-01/mms"])

    entity_data = []

    for entity_uri, entity_future in tqdm(
        download_entities(
            unvisited_entities,
            visited_entities,
            max_entities,
            max_parallel_requests,
            token,
        ),
        total=max_entities,
    ):
        entity = entity_future.result()
        unvisited_entities.remove(entity_uri)
        visited_entities.add(entity_uri)

        if "child" in entity:
            for child in entity["child"]:
                if child not in visited_entities:
                    unvisited_entities.add(child)

        entity_data.append(entity)

    print("Visited {} entities".format(len(visited_entities)))

    save_entities(entity_data)


def download_entities(
    unvisited_entities, visited_entities, max_entities, max_parallel_requests, token
):
    max_threads = 16

    thread_pool = ThreadPoolExecutor(max_workers=max_threads)

    while len(visited_entities) < max_entities:
        if len(unvisited_entities) == 0:
            print("No more entities to visit")
            break

        sample_size = min(max_parallel_requests, len(unvisited_entities))

        entity_uris = random.sample(sorted(unvisited_entities), sample_size)

        entity_futures = {}

        for entity_uri in entity_uris:
            entity_futures[entity_uri] = thread_pool.submit(
                download_entity, entity_uri, token
            )

        for entity_uri, entity_future in entity_futures.items():
            yield entity_uri, entity_future


def download_entity(entity_uri, token):
    print("Visiting entity: {}".format(entity_uri))

    # HTTP header fields to set
    headers = {
        "Authorization": "Bearer " + token,
        "Accept": "application/json",
        "Accept-Language": "en",
        "API-Version": "v2",
    }

    request = requests.get(entity_uri, headers=headers, verify=False)

    entity = request.json()

    return entity


def get_access_token():
    token_endpoint = "https://icdaccessmanagement.who.int/connect/token"
    client_id = (
        "cf8ad7e4-8d98-4064-86f0-c32e98197fa9_73cbc95d-f750-49b2-96ca-37ea9cc2bdc3"
    )
    client_secret = "U3VF2q06kJW00OVTLPLjBvDD4hchxAUYPlB1qQhWL64="
    scope = "icdapi_access"
    grant_type = "client_credentials"

    # get the OAUTH2 token
    payload = {
        "client_id": client_id,
        "client_secret": client_secret,
        "scope": scope,
        "grant_type": grant_type,
    }

    r = requests.post(token_endpoint, data=payload, verify=False).json()
    token = r["access_token"]

    return token


def save_entities(entity_data):
    with jsonlines.open("entities.jsonl", mode="w") as writer:
        writer.write_all(entity_data)


main()
