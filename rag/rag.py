import llama

llm = RetrievalAugmentedRunner(
    config={'chunk_size'=512,
            'step_size'=512,
            'batch_size'=512,
            'production':
            #{'url': 'https://api.staging.powerml.co',
            #'key': 'e197d0cafe85e6fd5feee10c8d212d3abb5b382f'}}
            {'url': 'https://api.powerml.co',
             'key': 'c0387a9fe3fea6ba55826ae7a3c52d32d620f94b'}}
)
llm.load_data("my_data")
llm.train()
response = llm("What were the worst rated projects launched by my company in 2023?")
print(response)

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
    
    def load_data(self, path):
        self.loader = DirectoryLoader(
            path,
            batch_size=self.batch_size,
            chunker=DefaultChunker(
                chunk_size=self.chunk_size, step_size=self.step_size
            ),            
        )

    def train(self):
	self.index = LaminiIndex(self.loader, self.config)

    def __call__(self, query):
        query_engine = QueryEngine(
            self.index,
            k=self.k,
            model_name=self.model_name,
            config=self.config,
            system_prompt=self.system_prompt,
        )
        return query_engine.answer_question(query)        

class LaminiIndex:
    def __init__(self, loader=None, config={}):
        self.loader = loader
	self.config = config
        if loader is not None:
            self.build_index()

    def build_index(self):
        self.splits = []
        self.index = None

        # load a batch of splits from a generator             
        for split_batch in tqdm(self.loader):
            embeddings = self.get_embeddings(split_batch)
            if self.index is None:
                # initialize the index                     
                self.index = faiss.IndexFlatL2(len(embeddings[0]))
            # add the embeddings to the index
            self.index.add(embeddings)
            # save the splits   
            self.splits.extend(split_batch)                

class QueryEngine:
    def __init__(
        self,
        index,
        model_name="meta-llama/Llama-2-7b-chat-hf",
        config={},
        k=5,
        system_prompt=None,
    ):
        self.index = index
        self.model = LlamaV2Runner(
            model_name=model_name, config=config, system_prompt=system_prompt
        )
        self.k = k

    def answer_question(self, question):
        prompt = self._build_prompt(question)
        return self.model(prompt)

    def _build_prompt(self, question):
        most_similar = self.index.query(question, k=self.k)
        prompt = "\n".join(reversed(most_similar)) + "\n\n" + question

        return prompt    
