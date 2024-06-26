from lamini.generation.base_prompt_object import PromptObject
from lamini.generation.embedding_node import EmbeddingNode

from typing import Iterator, AsyncIterator, Optional, Union

from lamini_rag.lamini_pipeline import DatasetDescriptor

class LaminiEmbeddingModelStage(EmbeddingNode):
    """ 
    This child class of EmbeddingNode is for use within a
    GenerationPipeline (or a child of that class). The class
    extension here is adding functionality for a query prompt 
    getter method that is ready for async requests. 

    For more information on how EmbeddingNodes work, refer to 
        https://github.com/lamini-ai/lamini/blob/main/lamini/generation/embedding_node.py
        
    Parameters
    ----------
    dataset: DatasetDescriptor
        Helper class used for output format to be passed to 
        the API calls
        
    """

    def __init__(self, dataset: DatasetDescriptor):
        super().__init__()
        self.dataset = dataset

    def generate(
        self,
        prompt: Union[Iterator[PromptObject], AsyncIterator[PromptObject]],
        model_name: Optional[str] = None,
    ):
        """ Get the query prompt before passing to the parent class'
        generate function.

        Parameters
        ----------
        prompt: Union[Iterator[PromptObject], AsyncIterator[PromptObject]]
            An iterator (Async or not) used as the input for the async call
            of get_query_prompt to encapsulate the text for model input 
            within a PromptObject

        model_name: Optional[str]
            Model name that will be passed to the API call
        
        Returns
        -------
        Generator: 
            A generator is returned from the parent class' generate call, 
            which in turn is returned from this function. For more information
            on the generator returned, refer to:
                https://github.com/lamini-ai/lamini/blob/main/lamini/generation/generation_node.py#L48
                The parent EmbeddingNode of this class is itself a child of the GenerationNode,
                which is why you should refer to GenerationNode for parent functionality
        """
        prompt = self.get_query_prompt(prompt)

        return super().generate(
            prompt,
            model_name=model_name,
        )

    async def get_query_prompt(self, prompts):
        """ A generator function that will yield the PromptObject
        built for the provided prompts in the prompts Iterator

        Parameters
        ----------
        prompts: Union[Iterator[PromptObject], AsyncIterator[PromptObject]]
            An iterator (Async or not) used as the input for the async call
            of get_query_prompt to encapsulate the text for model input 
            within a PromptObject
        
        Yields
        -------
        PromptObject: 
            Object storing and building the query and associated relevant info
            for RAG inference
        """
        async for prompt in prompts:
            query = prompt.data.get_query()

            yield PromptObject(prompt=query, data=prompt.data)

