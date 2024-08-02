from typing import List, Dict, Any
import asyncio
import jsonlines
import logging

from argparse import ArgumentParser, Namespace
from tqdm import tqdm

from lamini.generation.base_prompt_object import PromptObject
from lamini.generation.generation_pipeline import GenerationPipeline
from lamini.generation.generation_node import GenerationNode


def setup_logging() -> None:
    """ Logging standardization at the DEBUG level

    Parameters
    ----------
    None

    Returns
    -------
    None
    """

    # Remove all handlers associated with the root logger object.
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler()],
    )


def parse_arguments() -> Namespace:
    """ Argument Parser setup 
    The following arguments are used in this test script:
        --max-examples
            Max number of examples to evaluate
            default 3
        
        --output-path
            This is the path that will be storing the output results
            when calling the pipeline with the supplied data from
            --test-data. Data will be stored in the jsonl format
            
    Parameters
    ----------
    None
            
    Returns
    -------
    argparse.Namespace
        Namespace object storing the given args as attributes
    """

    parser = ArgumentParser()

    # The max number of examples to evaluate
    parser.add_argument(
        "--max-examples",
        type=int,
        default=3,
        help="The max number of examples to evaluate",
    )

    # The output path to store the generated results
    parser.add_argument(
        "--output-path",
        type=str,
        default="/app/lamini-earnings-sdk/data/results/rag_spot_check_results.jsonl",
        help="Path for results of model generation to be stored",
    )

    return parser.parse_args()


async def run_spot_check(args) -> List[PromptObject]:
    """ Main runtime function to run the spot check for prompt tuning 

    Parameters
    ----------
    args: argparse.Namespace
        Namespace object storing user provided arguments

    Returns
    -------
    List[lamini.generation.base_prompt_object.PromptObject]
        PromptObjects returned from the SpotCheckPipeline are wrapped
        into a list
    """

    dataset = load_dataset(args)

    results = SpotCheckPipeline().call(dataset)

    result_list = []

    pbar = tqdm(desc="Saving results", unit=" results")
    async for result in results:
        result_list.append(result)
        pbar.update()

    return result_list


class DatasetDescriptor:
    def get_output_type(self):
        """ Logging standardization at the DEBUG level

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        return {"model_answer": "str"}


def load_dataset(args: Namespace) -> PromptObject:
    """Load in test dataset
    Load in the test data from the args.test_data attribute, expected
    format is to be in jsonl.

    Each data point in the test_data is then added into a EarningExample 
    object which is then wrapped into a PromptObject to be yielded back
    to the returning function

    Yield is used as this function is expected to be used within an async 
    call to the Lamini platform.

    Parameters
    ----------
    args: argparse.Namespace
        Namespace object storing
            test_data: path string to the evaluation data
            max_examples: limit of number of examples to yield

    Yields
    ------
    lamini.generation.base_prompt_object.PromptObject
        PromptObject storing the single EarningsExamples prompt and data
    """

    path = "/app/lamini-earnings-sdk/data/golden_test_set.jsonl"

    with jsonlines.open(path) as reader:
        dataset = list(reader)

    for index, data in enumerate(dataset):
        if index < args.max_examples:
            yield PromptObject(
                prompt="", data=data
            )


def make_prompt(example: Dict[str, Any]) -> str:
    """ Construct a prompt using a template and inject the
    specific example information and question into the prompt.

    Parameters
    ----------
    example: Dict[str, Any]
        Company example

    Returns
    -------
    prompt: str
        Formatted query with relevant information and question
    """

    prompt = "You are an expert analyst from Goldman Sachs with 15 years of experience."
    prompt += " Consider the following company: \n"
    prompt += "==========================\n"
    prompt += get_company_info(example)
    prompt += "==========================\n"
    prompt += "Answer the following question: \n"
    prompt += example["question"]
    return prompt


def get_company_info(example: Dict[str, Any]) -> str:
    """ Construct a string using the company information

    Parameters
    ----------
    example: Dict[str, Any]
        Company example

    Returns
    -------
    prompt: str
        Formatted query with relevant information and question
    """

    prompt = f"Date of the call: {example['date']}\n"
    prompt += f"Ticker: {example['ticker']}\n"
    prompt += f"Quarter: {example['q']}\n"

    return prompt


class LaminiModelStage(GenerationNode):
    def preprocess(self, prompt: PromptObject):
        new_prompt = "<|begin_of_text|><|start_header_id|>user<|end_header_id|>"
        new_prompt += make_prompt(prompt.data) + "<|eot_id|>"
        new_prompt += "<|start_header_id|>assistant<|end_header_id|>"
        prompt.prompt = new_prompt


class SpotCheckPipeline(GenerationPipeline):
    """ 
    An extension fo the GenerationPipeline. This
    class is a simple example showcasing how to build LLM
    pipelines using Lamini generation nodes.
    """

    def __init__(self):
        super().__init__()
        self.model_stage = LaminiModelStage(
            "meta-llama/Meta-Llama-3-8B-Instruct", max_new_tokens=150
        )

    def forward(self, x):
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
        x = self.model_stage(x, output_type={"model_answer": "str"})
        return x


def save_results(args, results) -> None:
    """Store the generated results into the provided output path

    Parameters
    ----------
    args: argparse.Namespace
        Namespace object storing
            output_path: path string for the output file

    results: List[lamini.generation.base_prompt_object.PromptObject]
        Results from the generation pipeline SpotCheckPipeline
    """

    with jsonlines.open(args.output_path, "w") as writer:
        for result in results:

            row = result.data["example"].example.copy()
            row["model_answer"] = result.response["model_answer"]
            row["prompt"] = result.prompt

            writer.write(row)

            print("\n\n")
            print(result.prompt, "\n")
            print("Model Answer: ", result.response["model_answer"])
            print("\n")

if __name__ == "__main__":
    # Run preprocess functions of initalizing logging and argument parsing 
    setup_logging()

    args = parse_arguments()

    # Run the spot check pipeline with the loaded (or newly built) index
    results = asyncio.run(run_spot_check(args))

    # Store results in --output-path
    save_results(args, results)
