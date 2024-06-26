from lamini.generation.base_prompt_object import PromptObject
from lamini.generation.embedding_node import EmbeddingNode

from typing import Iterator, AsyncIterator, Optional, Union

class LaminiEmbeddingModelStage(EmbeddingNode):
    def __init__(self, dataset):
        super().__init__()
        self.dataset = dataset

    def generate(
        self,
        prompt: Union[Iterator[PromptObject], AsyncIterator[PromptObject]],
        model_name: Optional[str] = None,
    ):

        prompt = self.get_query_prompt(prompt)

        return super().generate(
            prompt,
            model_name=model_name,
        )

    async def get_query_prompt(self, prompts):
        async for prompt in prompts:
            query = prompt.data.get_query()

            yield PromptObject(prompt=query, data=prompt.data)

