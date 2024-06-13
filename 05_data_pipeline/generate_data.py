from lamini.generation.generation_node import GenerationNode
from lamini.generation.generation_pipeline import GenerationPipeline
from lamini.generation.base_prompt_object import PromptObject

import jsonlines

import itertools
import asyncio
from tqdm import tqdm

from typing import Union, Iterator, AsyncIterator

import logging

logger = logging.getLogger(__name__)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)


async def main():
    earnings_calls = load_earnings_calls()

    answers = QuestionAnswerPipeline().call(earnings_calls)

    await save_answers(answers)


async def load_earnings_calls():
    path = "/app/lamini-earnings-sdk/data/test_set_transcripts.jsonl"

    with jsonlines.open(path) as reader:
        for line in itertools.islice(reader, 1):
            logger.info(f"Loaded earnings call for {line['ticker']}")
            yield PromptObject(prompt="", data=line)


class QuestionAnswerPipeline(GenerationPipeline):
    def __init__(self):
        super(QuestionAnswerPipeline, self).__init__()

        self.question_generator = QuestionGenerator()
        self.answer_generator = AnswerGenerator()

    def forward(self, x):
        x = self.question_generator(x)
        x = self.answer_generator(x)
        return x


class QuestionGenerator(GenerationNode):
    def __init__(self):
        super(QuestionGenerator, self).__init__(
            model_name="meta-llama/Meta-Llama-3-8B-Instruct", max_new_tokens=150
        )

    def generate(
        self,
        prompt: Union[Iterator[PromptObject], AsyncIterator[PromptObject]],
        *args,
        **kwargs,
    ):
        prompt = self.add_template(prompt)

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
            logger.debug(f"Generated question for {result}")
            if result is None:
                continue

            if "question_1" not in result.response:
                continue

            if "question_2" not in result.response:
                continue

            if "question_3" not in result.response:
                continue

            questions = (
                result.response["question_1"],
                result.response["question_2"],
                result.response["question_3"],
            )
            for question in questions:
                result = PromptObject(prompt=question, data=result.data.copy())
                yield result

    async def add_template(self, prompts):
        async for prompt in prompts:
            chunks = chunk_prompt(prompt)
            for chunk in chunks:
                chunk.prompt = self.make_prompt(chunk)
                logger.info(
                    f"Generating question for {chunk.data['ticker']}, {chunk.data['q']}"
                )
                yield chunk

    def make_prompt(self, chunk):
        prompt = "<|begin_of_text|><|start_header_id|>user<|end_header_id|>"

        prompt += (
            "You are a financial analyst with extensive experience at Goldman Sachs."
        )
        prompt += "You are reading the earnings call transcript for the following company:\n\n"
        prompt += "====================\n\n"
        prompt += get_company_info(chunk) + "\n"
        prompt += "====================\n\n"
        prompt += (
            "You are reading the following section of the earnings call transcript:\n\n"
        )
        prompt += "====================\n\n"
        prompt += chunk.data["transcript"]
        prompt += "====================\n\n"
        prompt += "Consider the numbers in the transscript. "
        prompt += "Ask three questions about the numbers in the transcript that require precise answers. "
        prompt += "Only ask questions that can be answered using the transcript."
        prompt += "<|eot_id|>"
        prompt += "<|start_header_id|>assistant<|end_header_id|>"

        return prompt


def chunk_prompt(prompt):
    transcript = prompt.data["transcript"]
    chunk_size = 4096
    chunk_step = 2048

    for i in range(0, len(transcript), chunk_step):
        chunk = transcript[i : i + chunk_size]
        chunked_data = prompt.data.copy()
        chunked_data["transcript"] = chunk
        prompt_object = PromptObject(prompt=prompt.prompt, data=chunked_data)

        yield prompt_object


def get_company_info(chunk):
    info = f"Company: {chunk.data['exchange']}\n"
    info += f"Ticker: {chunk.data['ticker']}\n"
    info += f"Date: {chunk.data['date']}\n"
    info += f"Quarter: {chunk.data['q']}\n"
    return info


class AnswerGenerator(GenerationNode):
    def __init__(self):
        super(AnswerGenerator, self).__init__(
            model_name="meta-llama/Meta-Llama-3-8B-Instruct", max_new_tokens=150
        )

    def generate(
        self,
        prompt: Union[Iterator[PromptObject], AsyncIterator[PromptObject]],
        *args,
        **kwargs,
    ):
        prompt = self.add_template(prompt)
        results = super(AnswerGenerator, self).generate(prompt, output_type={"answer" : "str"}, *args, **kwargs)
        return results

    async def process_results(self, results):
        async for result in results:
            logger.info(f"Generated answer for {result}")
            if result is None:
                continue
            yield result

    async def add_template(self, prompts):
        async for prompt in prompts:
            logger.info(
                f"Generating answer for {prompt.data['ticker']}, {prompt.data['q']}, {prompt.prompt}"
            )
            prompt.data["question"] = prompt.prompt
            prompt.prompt = self.make_prompt(prompt)
            yield prompt

    def make_prompt(self, chunk):
        prompt = "<|begin_of_text|><|start_header_id|>user<|end_header_id|>"

        prompt += (
            "You are a financial analyst with extensive experience at Goldman Sachs."
        )
        prompt += "You are reading the earnings call transcript for the following company:\n\n"
        prompt += "====================\n\n"
        prompt += get_company_info(chunk)
        prompt += "====================\n\n"
        prompt += (
            "You are reading the following section of the earnings call transcript:\n\n"
        )
        prompt += "====================\n\n"
        prompt += chunk.data["transcript"] + "\n"
        prompt += "====================\n\n"
        prompt += "Consider the numbers in the transcript. "
        prompt += "If the answer to the question cannot be found in the transcript, reply that you do not know. "
        prompt += "Answer the following questions about the numbers in the transcript. "
        prompt += chunk.prompt
        prompt += "<|eot_id|>"
        prompt += "<|start_header_id|>assistant<|end_header_id|>"

        return prompt


async def save_answers(answers):
    path = "/app/lamini-earnings-sdk/data/results/generated_q_a.jsonl"

    with jsonlines.open(path, "w") as writer:
        pbar = tqdm(desc="Saving answers", unit=" answers")
        async for answer in answers:
            answer = {
                "ticker": answer.data["ticker"],
                "q": answer.data["q"],
                "date": answer.data["date"],
                "transcript": answer.data["transcript"],
                "prompt": answer.prompt,
                "question": answer.data["question"],
                "answer": answer.response["answer"],
            }
            writer.write(answer)
            pbar.update()


asyncio.run(main())
