
import jsonlines


class EarningsCallLoader:
    def __init__(self, path):
        self.path = path
        self.batch_size = 128
        self.limit = 100
        self.embedding_size=384
        self.chunker = EarningsCallChunker()

    def get_chunks(self):
        return self.chunker.get_chunks(self.load())

    def get_chunk_batches(self):
        # A generator that yields batches of chunks
        # Each batch is a list of strings, each a substring of the text with length self.batch_size
        # the last element of the list may be shorter than self.batch_size
        chunks = []
        for chunk in self.get_chunks():
            chunks.append(chunk)
            if len(chunks) == self.batch_size:
                yield chunks
                chunks = []

        if len(chunks) > 0:
            yield chunks

    def load(self):
        with jsonlines.open(self.path) as reader:
            for index, obj in enumerate(reader):
                yield obj["transcript"], obj["ticker"]

                if index >= self.limit:
                    break

    def __iter__(self):
        return self.get_chunk_batches()

    def __len__(self):
        return len(list(self.get_chunks()))


class EarningsCallChunker:
    def __init__(self, chunk_size=2048, step_size=512):
        self.chunk_size = chunk_size
        self.step_size = step_size

    def get_chunks(self, data):
        # return a list of strings, each a substring of the text with length self.chunk_size
        # the last element of the list may be shorter than self.chunk_size
        for text, ticker in data:
            for i in range(0, len(text), self.step_size):
                max_size = min(self.chunk_size, len(text) - i)

                prompt = "====================\n"
                prompt += (
                    "This is a section from an earnings call about company with ticker: "
                    + ticker
                    + "\n"
                )
                prompt += text[i : i + max_size]
                prompt += "\n"
                prompt += "====================\n"

                yield prompt

                if i + max_size == len(text):
                    break
