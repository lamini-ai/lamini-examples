from lamini_rag.lamini_index import LaminiIndex

from lamini.generation.generation_node import GenerationNode
from lamini.generation.base_prompt_object import PromptObject

from typing import Union, Iterator, AsyncIterator


class LaminiRAGModelStage(GenerationNode):
    """ 
    This child class of GenerationNode is for use within a
    GenerationPipeline (or a child of that class). The class
    extension here is overwriting the preprocess function 
    in order to query the vector index and embed those results
    into the prompt before calling generate. 

    For more information on how GenerationNode work, refer to 
        https://github.com/lamini-ai/lamini/blob/main/lamini/generation/generation_node.py
    
    Parameters
    ----------
    model_path: str
        Location of the index model
    """

    def __init__(self, model_path: str):
        super().__init__(
            model_name="meta-llama/Meta-Llama-3-8B-Instruct",
            max_new_tokens=150,
        )
        self.index = LaminiIndex.load_index(model_path)

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

        query_embedding = prompt.response

        results = self.index.mmr_query(query_embedding, k=40, n=3)

        new_prompt = "<|begin_of_text|><|start_header_id|>user<|end_header_id|>"
        new_prompt += "Consider the following:\n\n"
        for result in results:
            new_prompt += result + "\n\n"
        new_prompt += prompt.data["example"].get_prompt() + "<|eot_id|>"
        new_prompt += "<|start_header_id|>assistant<|end_header_id|>"

        return PromptObject(prompt=new_prompt, data=prompt.data)
