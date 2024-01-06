from directory_loader import DirectoryLoader, DefaultChunker
from lamini_index import LaminiIndex
from query_engine import QueryEngine


class RetrievalAugmentedRunner:
    def __init__(
        self,
        model_name="mistralai/Mistral-7B-Instruct-v0.1",
        config={},
        k=5,
        chunk_size=512,
        step_size=128,
        batch_size=512,
        system_prompt=None,
    ):
        self.config = config
        self.model_name = model_name

        self.k = k
        self.chunk_size = chunk_size
        self.step_size = step_size
        self.batch_size = batch_size
        self.system_prompt = system_prompt

    def load_data(self, path, exclude_patterns):
        self.loader = DirectoryLoader(
            path,
            batch_size=self.batch_size,
            chunker=DefaultChunker(
                chunk_size=self.chunk_size, step_size=self.step_size
            ),
            exclude_patterns=exclude_patterns,
        )

    def train(self):
        self.index = LaminiIndex(self.loader, self.config)

    def __call__(self, query):
        return self.call(query)

    def call(self, query):        
        query_engine = QueryEngine(
            self.index,
            k=self.k,
            model_name=self.model_name,
            config=self.config,
            system_prompt=self.system_prompt,
        )
        return query_engine.answer_question(query)
