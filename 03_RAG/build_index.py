from lamini_index import LaminiIndex

import jsonlines


def main():
    loader = ICDEntityLoader(
        path="/app/lamini-rag/data/best_entities_with_descriptions.jsonl"
    )

    index = LaminiIndex(loader=loader)

    index.save_index("/app/lamini-rag/models")


class ICDEntityLoader:
    def __init__(self, path):
        self.path = path
        self.batch_size = 128

    def __iter__(self):
        batch = []
        with jsonlines.open(self.path) as reader:
            for obj in reader:
                batch.append(obj["description"])
                if len(batch) == self.batch_size:
                    yield batch
                    batch = []

        if len(batch) > 0:
            yield batch



main()
