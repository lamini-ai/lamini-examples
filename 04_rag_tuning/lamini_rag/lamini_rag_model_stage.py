from lamini_rag.lamini_index import LaminiIndex

from lamini.generation.generation_node import GenerationNode
from lamini.generation.base_prompt_object import PromptObject

from typing import Union, Iterator, AsyncIterator


class LaminiRAGModelStage(GenerationNode):
    def __init__(self):
        super().__init__(
            model_name="meta-llama/Meta-Llama-3-8B-Instruct",
            max_new_tokens=150,
        )
        model_path = "/app/lamini-earnings-sdk/04_rag_tuning/rag_model"
        self.index = LaminiIndex.load_index(model_path)

    def preprocess(self, prompt: PromptObject):
        query_embedding = prompt.response

        results = self.index.mmr_query(query_embedding, k=40, n=3)

        new_prompt = "<|begin_of_text|><|start_header_id|>user<|end_header_id|>"
        new_prompt += "Consider the following:\n\n"
        for result in results:
            new_prompt += result + "\n\n"
        new_prompt += prompt.data["example"].get_prompt() + "<|eot_id|>"
        new_prompt += "<|start_header_id|>assistant<|end_header_id|>"

        return PromptObject(prompt=new_prompt, data=prompt.data)
