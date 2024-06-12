from lamini.generation.generation_node import GenerationNode
from lamini.generation.base_prompt_object import PromptObject

from typing import Union, Iterator, AsyncIterator


class LaminiModelStage(GenerationNode):
    def __init__(self, dataset, model_name="meta-llama/Meta-Llama-3-8B-Instruct",):
        super().__init__(
            model_name=model_name,
            max_new_tokens=150,
        )
        self.model_name = model_name
        self.dataset = dataset

    def generate(
        self,
        prompt: Union[Iterator[PromptObject], AsyncIterator[PromptObject]],
        *args,
        **kwargs,
    ):
        prompt = self.add_template(prompt)

        results = super().generate(
            prompt,
            model_name=self.model_name,
            output_type=self.dataset.get_output_type(),
            *args,
            **kwargs,
        )

        return results

    async def add_template(self, prompts):
        async for prompt in prompts:

            new_prompt = "<|begin_of_text|><|start_header_id|>user<|end_header_id|>"
            new_prompt += prompt.data.get_prompt() + "<|eot_id|>"
            new_prompt += "<|start_header_id|>assistant<|end_header_id|>"

            yield PromptObject(prompt=new_prompt, data=prompt.data)
