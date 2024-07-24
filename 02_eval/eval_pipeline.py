import asyncio
import logging

from tqdm import tqdm
from typing import AsyncIterator, Iterator, Union

from lamini.generation.base_prompt_object import PromptObject
from lamini.generation.generation_node import GenerationNode
from lamini.generation.generation_pipeline import GenerationPipeline
from lamini.generation.modify_node import ModifyNode


logger = logging.getLogger(__name__)


def evaluate_model(dataset, args):

    results = asyncio.run(run_evaluation_pipeline(dataset, args))

    print("Total results:", len(results))
    print(
        "Avg precision score:",
        sum([result.data["result"]["is_exact_match"] for result in results])
        / len(results),
    )
    print(
        "Avg score:",
        sum([result.data["result"]["score"] for result in results]) / len(results),
    )

    return results


async def run_evaluation_pipeline(dataset, args):
    results = EvaluationPipeline().call(dataset)

    result_list = []

    pbar = tqdm(desc="Saving results", unit=" results")
    async for result in results:
        result_list.append(result)
        pbar.update()

    return result_list


class EvaluationPipeline(GenerationPipeline):
    def __init__(self):
        super().__init__()

        self.model_gen_stage = LaminiModelStage()
        self.modify_stage = ModifyStage()
        self.score_stage = ScoreStage()

    def forward(self, x):
        x = self.model_gen_stage(x, output_type={
            "answer": "str",
            "value": "float",
            "units": "str",
        })
        x = self.modify_stage(x)
        x = self.score_stage(x, output_type={
            "explanation": "str",
            "score": "int",
        })
        return x


class LaminiModelStage(GenerationNode):
    def __init__(self):
        super().__init__(
            model_name="meta-llama/Meta-Llama-3-8B-Instruct",
            max_new_tokens=150,
        )

    def preprocess(self, prompt: PromptObject):
        example = prompt.data["example"]
        new_prompt = "<|begin_of_text|><|start_header_id|>user<|end_header_id|>"
        new_prompt += example.get_prompt() + "<|eot_id|>"
        new_prompt += "<|start_header_id|>assistant<|end_header_id|>"
        return PromptObject(prompt=new_prompt, data=prompt.data)


class ModifyStage(ModifyNode):
    def __init__(self):
        super().__init__(self.modify_result)

    def modify_result(self, result: PromptObject):
        result.data["example"].response = result.response


class ScoreStage(GenerationNode):
    def __init__(self):
        super().__init__(
            model_name="meta-llama/Meta-Llama-3-8B-Instruct",
            max_new_tokens=150,
        )

    def preprocess(self, example):
        response = example.data["example"].format_response(example.response)

        prompt = "<s>[INSTR]A large language model (LLM) is going to answer a question. "
        prompt += (
            "Your job is to score the answer, comparing it to a golden reference. "
        )
        prompt += "You are an expert scorer.\n\n"
        prompt += "Rate the answer using a score from 1 (lowest match) to 5 (highest match).\n"
        prompt += example.data["example"].get_rubric()
        prompt += "Use the full range. Read the gold answer carefully. "
        prompt += "Explain your score in 2-3 sentences, then assign a score. "
        prompt += 'Output your score as a JSON object in the format {"explanation" : str, "score" : int}\n'
        prompt += "Use single quotes within your explanation. End your explanation with a double quote.\n"
        prompt += "Prefer answers that are most similar to the gold answer, even if the gold answer refused to answer the question.\n\n"
        prompt += f"========== question =========\n{example.data['example'].get_question()}\n\n"
        prompt += f"========== gold answer =========\n{example.data['example'].get_response(response)}\n\n"
        prompt += f"========== model answer =========\n{response}\n\n"
        prompt += "=" * 40 + "\n\n"
        prompt += f"How would you score the model's answer compared to the gold answer (using the 1-5 scale defined above)?[/INSTR]"

        example.prompt = prompt

    def postprocess(self, result):
        result.data["result"] = {
            "example_id": result.data["example"].get_id(),
            "prompt": result.data["example"].get_prompt(),
            "response": result.data["example"].response,
            "reference_response": result.data["example"].get_response_json(),
            "is_exact_match": result.data["example"].is_exact_match(result.data["example"].response),
            "score": result.response["score"],
            "explanation": result.response["explanation"],
        }
