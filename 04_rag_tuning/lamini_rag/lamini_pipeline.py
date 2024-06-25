from lamini_rag.lamini_embedding_model_stage import LaminiEmbeddingModelStage
from lamini_rag.lamini_rag_model_stage import LaminiRAGModelStage

from lamini.generation.generation_pipeline import GenerationPipeline


class DatasetDescriptor:
    def get_output_type(self):
        return {"model_answer": "str"}


class SpotCheckPipeline(GenerationPipeline):
    def __init__(self, dataset):
        super().__init__()
        self.embedding_stage = LaminiEmbeddingModelStage(dataset)
        self.model_stage = LaminiRAGModelStage(dataset)

    def forward(self, x):
        x = self.embedding_stage(x)
        x = self.model_stage(x)
        return x