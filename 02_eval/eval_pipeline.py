import asyncio
import logging

from tqdm import tqdm
from typing import List, Any, AsyncGenerator, Generator, Union

from lamini.generation.base_prompt_object import PromptObject
from lamini.generation.generation_node import GenerationNode
from lamini.generation.generation_pipeline import GenerationPipeline
from lamini.generation.modify_node import ModifyNode

from load_earnings_call_dataset import EarningsCallsDataset

logger = logging.getLogger(__name__)


def evaluate_model(dataset: EarningsCallsDataset) -> List[Any]:
    """ Run model evaluation with the provided dataset

    Parameters
    ----------
    dataset: EarningsCallsDataset
        Object hanlding the loading and formatting of the jsonlines
        example data

    args: Namespace
        Input args at runtime

    Returns
    -------
    results: List[Any]
        Returned results from the evaluation pipline
    """

    results = asyncio.run(run_evaluation_pipeline(dataset))

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


async def run_evaluation_pipeline(dataset: EarningsCallsDataset) -> List[Any]:
    """ Run model evaluation with the provided dataset

    Parameters
    ----------
    dataset: EarningsCallsDataset
        Object hanlding the loading and formatting of the jsonlines
        example data

    Returns
    -------
    result_list: List[Any]
        Returned results from the evaluation pipline
    """

    results = EvaluationPipeline().call(dataset)

    result_list = []

    pbar = tqdm(desc="Saving results", unit=" results")
    async for result in results:
        result_list.append(result)
        pbar.update()

    return result_list


class EvaluationPipeline(GenerationPipeline):
    """
    Extension of a GenerationPipeline to generate, modify, then
    score the returned results from Lamini.generate.

    Parameters
    ----------
    None

    """

    def __init__(self) -> None:
        super().__init__()

        self.model_gen_stage = LaminiModelStage()
        self.modify_stage = ModifyStage()
        self.score_stage = ScoreStage()

    def forward(
            self, x: Union[Generator[PromptObject, None, None], AsyncGenerator[PromptObject, None]]
            ) -> AsyncGenerator[PromptObject, None]:
        """ Implementation of a forward call for this pipeline

        Parameters
        ----------
        x: Union[Generator[PromptObject, None, None], AsyncGenerator[PromptObject, None]]
            Provided input for pipeline

        Returns
        -------
        x: AsyncGenerator[PromptObject, None]
            Returned output from pipeline execution
        """

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
    """
    Extension of a GenerationNode for generation calls within a pipeline

    Parameters
    ----------
    None

    """

    def __init__(self):
        super().__init__(
            model_name="meta-llama/Meta-Llama-3.1-8B-Instruct",
            max_new_tokens=150,
        )

    def preprocess(self, prompt: PromptObject) -> PromptObject:
        """ Formatting of the prompt for Llama3 text markers

        Parameters
        ----------
        prompt: PromptObject
            Provided prompt for node

        Returns
        -------
        PromptObject
            Formatted new prompt ready for forward call
        """

        example = prompt.data["example"]
        new_prompt = "<|begin_of_text|><|start_header_id|>user<|end_header_id|>"
        new_prompt += example.get_prompt() + "<|eot_id|>"
        new_prompt += "<|start_header_id|>assistant<|end_header_id|>"
        return PromptObject(prompt=new_prompt, data=prompt.data)


class ModifyStage(ModifyNode):
    """
    Extension of a ModifyNode for generation calls within a pipeline

    Parameters
    ----------
    None

    """

    def __init__(self):
        super().__init__(self.modify_result)

    def modify_result(self, result: PromptObject) -> None:
        """ Data formatting for the provided result

        Parameters
        ----------
        result: PromptObject
            Provided result from a prior node in the pipeline

        Returns
        -------
        None
        """

        result.data["example"].response = result.response


class ScoreStage(GenerationNode):
    """
    Extension of a GenerationNode for scoring of a prompt within a pipeline

    Parameters
    ----------
    None

    """

    def __init__(self):
        super().__init__(
            model_name="meta-llama/Meta-Llama-3.1-8B-Instruct",
            max_new_tokens=150,
        )

    def preprocess(self, example: PromptObject) -> None:
        """ Preprocess provided prompt object before generate call

        Parameters
        ----------
        example: PromptObject
            Prompt object for which to preprocess

        Returns
        -------
        None
        """

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

    def postprocess(self, result: PromptObject) -> None:
        """ Postprocess provided prompt object after generate call

        Parameters
        ----------
        result: PromptObject
            Prompt object for which to postprocess

        Returns
        -------
        None
        """

        result.data["result"] = {
            "example_id": result.data["example"].get_id(),
            "prompt": result.data["example"].get_prompt(),
            "response": result.data["example"].response,
            "reference_response": result.data["example"].get_response_json(),
            "is_exact_match": result.data["example"].is_exact_match(result.data["example"].response),
            "score": result.response["score"],
            "explanation": result.response["explanation"],
        }
