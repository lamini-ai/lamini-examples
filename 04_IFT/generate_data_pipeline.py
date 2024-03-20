from lamini.generation.generation_node import GenerationNode
from lamini.generation.generation_pipeline import GenerationPipeline
from lamini.generation.base_prompt_object import PromptObject

import jsonlines

import asyncio
from tqdm import tqdm

from typing import Union, Iterator, AsyncIterator
import lamini
import logging

logger = logging.getLogger(__name__)

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

max_examples = 2
model_name = "mistralai/Mistral-7B-Instruct-v0.2"
lamini.api_key = ""

async def load_icd_dataset():
    i = 0
    path = "/app/lamini-ift/data/best_entities_with_descriptions.jsonl"
    with jsonlines.open(path) as reader:
        for line in reader:
            if i == max_examples:
                break
            i += 1
            yield PromptObject(prompt="", data=line)


async def main():
    dataset = load_icd_dataset()
    answers = ICDPipeline().call(dataset)

    await save_answers(answers)


class ICDPipeline(GenerationPipeline):
    def __init__(self):
        super(ICDPipeline, self).__init__()

        self.question_generator = QuestionGenerator()
        self.answer_generator = AnswerGenerator()

    def forward(self, x):
        x = self.question_generator(x)
        x = self.answer_generator(x)
        return x


class QuestionGenerator(GenerationNode):
    def __init__(self):
        super(QuestionGenerator, self).__init__(
            model_name=model_name
        )

    def generate(
            self,
            prompt: Union[Iterator[PromptObject], AsyncIterator[PromptObject]],
            *args,
            **kwargs,
    ):
        results = super(QuestionGenerator, self).generate(
            prompt,
            output_type={
                "question_1": "string",
                "question_2": "string",
                "question_3": "string",
            },
            *args,
            **kwargs,
        )
        return results

    async def process_results(self, results):
        async for result in results:
            if (result is None) or (result.response is None):
                continue
            if "code" not in result.data:
                print("\n======\nNo ICD11 code found in this record. Skipping.\n======")
                continue
            response = result.response
            questions = [response["question_1"], response["question_2"], response["question_3"]]
            for question in questions:
                ans = PromptObject(prompt=question, data=result.data.copy())
                yield ans

    async def transform_prompt(self, prompts):
        async for prompt in prompts:
            prompt.prompt = self.make_prompt(prompt)
            yield prompt

    def make_prompt(self, chunk):
        prompt = (
            "<s>[INST] You are a medical coding expert who has just read the recent ICD11 standard from the World Health Organization."
        )
        prompt += "You are going to read through a page from the standard carefully and generate three questions about it."
        prompt += "====================\n\n"
        prompt += chunk.data["description"]
        prompt += "====================\n\n"
        prompt += " [/INST]"
        return prompt


class AnswerGenerator(GenerationNode):
    def __init__(self):
        super(AnswerGenerator, self).__init__(
            model_name=model_name
        )

    # can be commented out as it has the base implementation as the node
    # def generate(
    #         self,
    #         prompt: Union[Iterator[PromptObject], AsyncIterator[PromptObject]],
    #         *args,
    #         **kwargs,
    # ):
    #     # prompts = self.transform_prompt(prompt)
    #     results = super(AnswerGenerator, self).generate(prompt, *args, **kwargs)
    #     return results
    #     # processed_results = self.process_results(results)
    #     # return processed_results

    async def process_results(self, results):
        async for result in results:
            logger.info(f"Generated answer for {result}")
            if result is None:
                continue
            yield result

    async def transform_prompt(self, prompts):
        async for prompt in prompts:
            prompt.data["question"] = prompt.prompt
            prompt.prompt = self.make_prompt(prompt)
            yield prompt

    def make_prompt(self, chunk):
        # llama 2 prompt <s>[INST] <<SYS>>\n{system}\n<</SYS>>\n\n{user} [/INST]
        prompt = (
            "<s>[INST] You are a medical coding expert who has just read the recent ICD11 standard from the World Health Organization."
        )
        prompt += "You are going to read through a page from the standard carefully, and answer a question about it.\n\n"
        prompt += chunk.data["description"] + "\n\n"
        prompt += "Now answer the following question:\n\n"
        prompt += chunk.data["question"]
        prompt += " [/INST]"
        return prompt


async def save_answers(answers):
    path = "/app/lamini-ift/data/icd_pipeline_qna.jsonl"

    with jsonlines.open(path, "w") as writer:
        pbar = tqdm(desc="Saving answers", unit=" answers")
        async for answer in answers:
            answer = {
                "question": answer.data["question"],
                "entity": answer.data,
                "answer": answer.response["output"]
            }
            writer.write(answer)
            pbar.update()


asyncio.run(main())
