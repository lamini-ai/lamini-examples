from typing import Union, AsyncIterator, Iterator

from lamini_rag.lamini_embedding_model_stage import LaminiEmbeddingModelStage
from lamini_rag.lamini_rag_model_stage import LaminiRAGModelStage
from lamini.generation.generation_pipeline import GenerationPipeline


class SpotCheckPipeline(GenerationPipeline):
    """ 
    An extension fo the GenerationPipeline that will put an
    embedding stage and a RAG model stage in sequence. This
    class is a simple example showcasing how to build LLM
    pipelines using Lamini generation nodes.
    """
    def __init__(self):
        super().__init__()
        self.embedding_stage = LaminiEmbeddingModelStage()
        self.model_stage = LaminiRAGModelStage()

    def forward(self, x: Union[Iterator, AsyncIterator]) -> AsyncIterator:
        """ Main function for execution of a provided prompt. This 
        is not intended to be the public function for running a prompt
        within a pipeline. This is a override of the function within
        the parent class here:
            https://github.com/lamini-ai/lamini/blob/main/lamini/generation/generation_pipeline.py

        Pipelines are intended to be called for execution of prompts. For example,
        the following line within the run_spot_check function:
            results = SpotCheckPipeline().call(dataset)

        Parameters
        ----------
        x: Union[Iterator, AsyncIterator]
            Iterator, or generators are passed between nodes and pipelines. This
            is the prompts being passed through into the corresponding stages of 
            the pipelines.
            See the call function within the generation pipeline to see what is
            being passed to the child function
            https://github.com/lamini-ai/lamini/blob/main/lamini/generation/generation_pipeline.py#L42
        
        Returns
        -------
        x: Generator
            The generator outputs from the final stage is returned.  
            See the call function within the generation node class for more information
            on what is returned from each stage:
            https://github.com/lamini-ai/lamini/blob/main/lamini/generation/generation_node.py#L42
        """
        x = self.embedding_stage(x)
        x = self.model_stage(x, output_type={"model_answer": "str"})
        return x