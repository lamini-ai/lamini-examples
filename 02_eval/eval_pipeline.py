from lamini.generation.base_prompt_object import PromptObject
from lamini.generation.generation_node import GenerationNode
from lamini.generation.generation_pipeline import GenerationPipeline
from lamini.generation.modify_node import ModifyNode

from tqdm import tqdm

import asyncio

from typing import AsyncIterator, Iterator, Union

import logging

logger = logging.getLogger(__name__)


def evaluate_model(model, dataset, args):

    results = asyncio.run(run_evaluation_pipeline(model, dataset, args))

    print("Total results:", len(results))
    print(
        "Avg precision score:",
        sum([result.data.result["is_exact_match"] for result in results])
        / len(results),
    )
    print(
        "Avg score:",
        sum([result.data.result["score"] for result in results]) / len(results),
    )

    return results


async def run_evaluation_pipeline(model, dataset, args):
    data_slice = slice_dataset(dataset, args)

    results = EvaluationPipeline(model, dataset).call(data_slice)

    result_list = []

    pbar = tqdm(desc="Saving results", unit=" results")
    async for result in results:
        result_list.append(result)
        pbar.update()

    return result_list


async def slice_dataset(dataset, args):
    for index, example in enumerate(dataset):
        if index < args.max_examples:
            yield PromptObject(prompt=example.get_prompt(), data=example)


class EvaluationPipeline(GenerationPipeline):
    def __init__(self, model, dataset):
        super().__init__()
        self.model_stages = model.get_stages(dataset)
        self.modify_stage = ModifyStage()
        self.score_stage = ScoreStage()

    def forward(self, x):
        for stage in self.model_stages:
            x = stage(x)

        x = self.modify_stage(x)
        x = self.score_stage(x)
        return x


class ModifyStage(ModifyNode):
    def __init__(self):
        super().__init__()

    async def modify(self, results):
        async for result in results:
            # filter out results that are None
            if result.response is None:
                logging.error(
                    f"Error evaluating example {result.data.get_id()}: {result.error}"
                )
                result.response = example.get_default_response()

            result.data.response = result.response

            yield result


class ScoreStage(GenerationNode):
    def __init__(self):
        super().__init__(
            model_name="meta-llama/Meta-Llama-3-8B-Instruct",
            max_new_tokens=150,
        )

    def generate(
        self,
        prompt: Union[Iterator[PromptObject], AsyncIterator[PromptObject]],
        *args,
        **kwargs,
    ):
        results = super().generate(
            prompt,
            output_type={"explanation": "str", "score": "int"},
            *args,
            **kwargs,
        )

        return results

    async def transform_prompt(self, examples):
        async for example in examples:
            example.prompt = self.make_prompt(example)

            yield example

    async def process_results(self, results):
        async for result in results:
            # filter out results that are None
            if result is None:
                continue

            if result.response is None:
                logging.error(
                    f"Error scoring example {result.data.get_id()}: {result.error}"
                )
                continue

            result.data.result = {
                "example_id": result.data.get_id(),
                "prompt": result.data.get_prompt(),
                "response": result.data.response,
                "reference_response": result.data.get_response_json(),
                "is_exact_match": result.data.is_exact_match(result.data.response),
                "score": result.response["score"],
                "explanation": result.response["explanation"],
            }

            yield result

    def make_prompt(self, example):
        response = example.data.format_response(example.response)

        prompt = "<s>[INSTR]A large language model (LLM) is going to answer a question. "
        prompt += (
            "Your job is to score the answer, comparing it to a golden reference. "
        )
        prompt += "You are an expert scorer.\n\n"
        prompt += "Rate the answer using a score from 1 (lowest match) to 5 (highest match).\n"
        prompt += example.data.get_rubric()
        prompt += "Use the full range. Read the gold answer carefully. "
        prompt += "Explain your score in 2-3 sentences, then assign a score. "
        prompt += 'Output your score as a JSON object in the format {"explanation" : str, "score" : int}\n'
        prompt += "Use single quotes within your explanation. End your explanation with a double quote.\n"
        prompt += "Prefer answers that are most similar to the gold answer, even if the gold answer refused to answer the question.\n\n"
        prompt += f"========== question =========\n{example.data.get_question()}\n\n"
        prompt += f"========== gold answer =========\n{example.data.get_response(response)}\n\n"
        prompt += f"========== model answer =========\n{response}\n\n"
        prompt += "=" * 40 + "\n\n"
        prompt += f"How would you score the model's answer compared to the gold answer (using the 1-5 scale defined above)?[/INSTR]"

        return prompt
