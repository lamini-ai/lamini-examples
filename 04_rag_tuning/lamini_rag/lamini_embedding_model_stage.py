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

    def preprocess(self, prompt: PromptObject) -> PromptObject:
        """ Construct a new prompt object given the prompt
        data. Preprocess is called before passing the prompt 
        to the generate, allowing for precise control for what 
        prompt adjustments are needed for this particular node.

        Parameters
        ----------
        prompt: PromptObject
            Prompt within the GenerationPipeline

        Returns
        -------
        PromptObject
            Newly instantiate prompt object
        """

        query = prompt.data["example"].get_query()
        return PromptObject(prompt=query, data=prompt.data)

