from typing import Union, Iterator, AsyncIterator, Generator, Dict, Any, AsyncGenerator
import asyncio
import itertools
import jsonlines
import logging

from tqdm import tqdm

from lamini.generation.generation_node import GenerationNode
from lamini.generation.generation_pipeline import GenerationPipeline
from lamini.generation.base_prompt_object import PromptObject

logger = logging.getLogger(__name__)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)


class QuestionAnswerPipeline(GenerationPipeline):
    """
    An extension fo the GenerationPipeline that will put two
    nodes in sequence, one to generate questions from a provided
    prompt, and the next to answer the generated question from the
    prior node.
    """

    def __init__(self):
        super(QuestionAnswerPipeline, self).__init__()

        self.question_generator = QuestionGenerator()
        self.answer_generator = AnswerGenerator()

    def forward(self, x: Union[Iterator, AsyncIterator]) -> AsyncIterator:
        """ Main function for execution of a provided prompt. This
        is not intended to be the public function for running a prompt
        within a pipeline. This is a override of the function within
        the parent class here:
            https://github.com/lamini-ai/lamini/blob/main/lamini/generation/generation_pipeline.py

        Pipelines are intended to be called for execution of prompts. For example,
        the following line within the run_spot_check function:
            results = SpotCheckPipeline().call(dataset)

        Parameters
        ----------
        x: Union[Iterator, AsyncIterator]
            Iterator, or generators are passed between nodes and pipelines. This
            is the prompts being passed through into the corresponding stages of
            the pipelines.
            See the call function within the generation pipeline to see what is
            being passed to the child function
            https://github.com/lamini-ai/lamini/blob/main/lamini/generation/generation_pipeline.py#L42

        Returns
        -------
        x: Generator
            The generator outputs from the final stage is returned.
            See the call function within the generation node class for more information
            on what is returned from each stage:
            https://github.com/lamini-ai/lamini/blob/main/lamini/generation/generation_node.py#L42
        """

        x = self.question_generator(x, output_type={
            "question_1": "str",
            "question_2": "str",
            "question_3": "str",
        })
        x = self.answer_generator(x)
        return x

def get_company_info(chunk: PromptObject) -> str:
    """Static function used for the GenerationNodes in the
    QuestionAnswerPipeline

    Parameters
    ----------
    chunk: PromptObject
        EarningsExample prompt data holding the
        company metadata

    Returns
    -------
    info: str
        Constructed string using the company metadata

    """
    info = f"Company: {chunk.data['exchange']}\n"
    info += f"Ticker: {chunk.data['ticker']}\n"
    info += f"Date: {chunk.data['date']}\n"
    info += f"Quarter: {chunk.data['q']}\n"
    return info


class QuestionGenerator(GenerationNode):
    """GenerationNode represents a step of processing in a pipeline.

    GenerationNode.__call__() is the entrypoint, which includes 3 sub-steps for each prompt:
    1. Transform prompt: invoke self.preprocess() to transform prompt to implement
       the semantic of this GenerationNode.
    2. Generate: invoke self.generate() (defined in GenerationNode) to send prompt
       to Lamini inference API, and fetch the response
    3. Transform response: invoke self.postprocess() to transform response to implement
       the semantic of this GenerationNode.

    If you do not need to transform prompt and response, you don't need to implement
    self.preprocess() or self.postprocess().

    You may implement self.preprocess() and self.postprocess() for your own purpose.
    """

    def __init__(self):
        super(QuestionGenerator, self).__init__(
            model_name="meta-llama/Meta-Llama-3.1-8B-Instruct", max_new_tokens=150
        )

    def preprocess(self, prompt: PromptObject) -> None:
        """ Log the prompt and prompt data that is being
        preprocessed within this Node.

        Parameters
        ----------
        prompt: PromptObject
            Prompt within the GenerationPipeline

        Returns
        -------
        None
        """

        prompt.prompt = self.make_prompt(prompt)
        logger.info(f"Generating question for {prompt.data['ticker']}, {prompt.data['q']}")

    def postprocess(self, prompt: PromptObject) -> Generator[PromptObject, None, None]:
        """ Postprocess the resulting prompts from the generate call
        of this Node. Iterate through all three questions generated
        and build new PromptObjects for each question.

        Parameters
        ----------
        prompt: PromptObject
            Prompt within the GenerationPipeline

        Yields
        ------
        ans: PromptObject
            New PromptObject that contains a single question
        """

        response = prompt.response
        questions = [
            response["question_1"],
            response["question_2"],
            response["question_3"],
        ]
        for question in questions:
            ans = PromptObject(prompt=question, data=prompt.data.copy())
            yield ans


    def make_prompt(self, obj: Dict[str, Any]) -> str:
        """ Construct a prompt using a template and inject the
        specific example information and question into the prompt.

        Parameters
        ----------
        obj: Dict[str, Any]
            Company example

        Returns
        -------
        prompt: str
            Formatted query with relevant information and question
        """

        prompt = (
            "<s>[INSTR]You are a financial analyst with extensive experience at Goldman Sachs."
        )
        prompt += "You are reading the earnings call transcript for the following company:\n\n"
        prompt += "====================\n\n"
        prompt += get_company_info(obj) + "\n"
        prompt += "====================\n\n"
        prompt += (
            "You are reading the following section of the earnings call transcript:\n\n"
        )
        prompt += "====================\n\n"
        prompt += obj.data["transcript"]
        prompt += "====================\n\n"
        prompt += "Consider the numbers in the transcript. "
        prompt += "Ask three questions about the numbers in the transcript that require precise answers. "
        prompt += "Only ask questions that can be answered using the transcript."
        prompt +="[/INSTR]"

        return prompt

class AnswerGenerator(GenerationNode):
    """GenerationNode used to answer the questions coming
    from the prior QuestionGenerator Node.
    """

    def __init__(self):
        super(AnswerGenerator, self).__init__(
            model_name="meta-llama/Meta-Llama-3.1-8B-Instruct", max_new_tokens=150
        )

    def postprocess(self, prompt: PromptObject) -> None:
        """ Postprocess the resulting prompts from the generate call
        of this Node. Iterate through all three questions generated
        and build new PromptObjects for each question.

        Parameters
        ----------
        prompt: PromptObject
            Prompt within the GenerationPipeline

        Returns
        -------
        None
        """

        logger.info(f"Generated answer for {prompt}")

    def preprocess(self, prompt: PromptObject) -> None:
        """ Construct a new prompt string given the prompt
        data

        Parameters
        ----------
        prompt: PromptObject
            Prompt within the GenerationPipeline

        Returns
        -------
        None
        """

        prompt.data["question"] = prompt.prompt
        prompt.prompt = self.make_prompt(prompt)

    def make_prompt(self, obj: PromptObject) -> str:
        """ Construct a prompt using a template and inject the
        specific example information and question into the prompt.

        Parameters
        ----------
        obj: Dict[str, Any]
            Company example

        Returns
        -------
        prompt: str
            Formatted query with relevant information and question
        """

        prompt = (
            "<s>[INSTR] You are a financial analyst with extensive experience at Goldman Sachs."
        )
        prompt += "You are reading the earnings call transcript for the following company:\n\n"
        prompt += "====================\n\n"
        prompt += get_company_info(obj)
        prompt += "====================\n\n"
        prompt += (
            "You are reading the following section of the earnings call transcript:\n\n"
        )
        prompt += "====================\n\n"
        prompt += obj.data["transcript"] + "\n"
        prompt += "====================\n\n"
        prompt += "Consider the numbers in the transcript. "
        prompt += "If the answer to the question cannot be found in the transcript, reply that you do not know. "
        prompt += "Answer the following questions about the numbers in the transcript. "
        prompt += obj.prompt
        prompt += "[/INSTR]"

        return prompt


async def load_earnings_calls() -> AsyncGenerator[PromptObject, None]:
    """ Load the test data set that is the earnings call curated
    responses the pipeline is tested against.

    Parameters
    ----------
    None

    Yields
    ------
    PromptObject
        Constructed prompt object from a single line within the test set.
    """

    path = "/app/lamini-earnings-sdk/data/test_set_transcripts.jsonl"

    with jsonlines.open(path) as reader:
        # Here we only read the 1st line from the .jsonl file.
        for line in itertools.islice(reader, 1):
            logger.info(f"Loaded earnings call for {line['ticker']}")
            yield PromptObject(prompt="", data=line)

async def save_answers(answers: Generator[PromptObject, None, None]) -> None:
    """Store the generated results into the provided output path

    Parameters
    ----------
    answers: Generator[PromptObject, None, None]
        Answers returned from the GenerationPipeline

    Returns
    -------
    None
    """

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
                "answer": answer.response["output"],
            }
            writer.write(answer)
            pbar.update()


async def run_pipeline() -> None:
    """Main execution function for this pipeline example

    Parameters
    ----------
    None

    Returns
    -------
    None
    """

    earnings_calls = load_earnings_calls()
    answers = QuestionAnswerPipeline().call(earnings_calls)
    await save_answers(answers)

if __name__ == "__main__":
    asyncio.run(run_pipeline())
