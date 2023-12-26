from rag.lamini_index import LaminiIndex

class RagRunner:
    def __init__(self, path):
        self.path = path
        self.index = LaminiIndex.load_index(path)

    def get_examples(self, query, k=5):
        chunks = self.index.query(query=query, k=k)

        return [self._chunk_to_example(chunk) for chunk in chunks]

    def _chunk_to_example(self, chunk):
        return {
            "task_parameters": "",
            "question": "",
            "answer": "",
            "context": chunk
        }
