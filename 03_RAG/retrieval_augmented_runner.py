from directory_loader import DirectoryLoader, DefaultChunker
from lamini_index import LaminiIndex
from query_engine import QueryEngine


class RetrievalAugmentedRunner:
    def __init__(
        self,
        model_name="mistralai/Mistral-7B-Instruct-v0.2",
        config={},
        k=5,
        chunk_size=512,
        step_size=128,
        batch_size=512,
    ):
        self.config = config
        self.model_name = model_name

        self.k = k
        self.chunk_size = chunk_size
        self.step_size = step_size
        self.batch_size = batch_size
        self.query_engine = None

    def load_data(self, path, exclude_files=[]):
        self.loader = DirectoryLoader(
            path,
            batch_size=self.batch_size,
            chunker=DefaultChunker(
                chunk_size=self.chunk_size, step_size=self.step_size
            ),
            exclude_files=exclude_files,
        )

    def train(self):
        self.index = LaminiIndex(self.loader, self.config)

    def load_index(self, path):
        self.index = LaminiIndex.load_index(path)

    def __call__(self, query):
        return self.call(query)

    def call(self, query):
        query_engine = QueryEngine(
            self.index,
            k=self.k,
            model_name=self.model_name,
            config=self.config,
        )
        return query_engine.answer_question(query)

    def query(self, query):
        self.query_engine = QueryEngine(
            self.index,
            k=self.k,
            model_name=self.model_name,
            config=self.config,
        )
        return self.query_engine.most_similar(query)

    def generate(self, query):
        return self.query_engine.generate(query)
