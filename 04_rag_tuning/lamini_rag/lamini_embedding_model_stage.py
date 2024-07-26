from lamini.generation.base_prompt_object import PromptObject
from lamini.generation.embedding_node import EmbeddingNode

class LaminiEmbeddingModelStage(EmbeddingNode):
    """ 
    This child class of EmbeddingNode is for use within a
    GenerationPipeline (or a child of that class). The class
    extension here is adding functionality for a query prompt 
    getter method that is ready for async requests. 

    For more information on how EmbeddingNodes work, refer to 
        https://github.com/lamini-ai/lamini/blob/main/lamini/generation/embedding_node.py
    """

    def preprocess(self, prompt: PromptObject):
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
        query = prompt.data["example"].get_query()
        return PromptObject(prompt=query, data=prompt.data)

