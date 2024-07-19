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
    def __init__(self):
        super(QuestionAnswerPipeline, self).__init__()

        self.question_generator = QuestionGenerator()
        self.answer_generator = AnswerGenerator()

    def forward(self, x):
        """Defines the pipeline
        
        forward() is invoked by __call__() as shown in run_pipeline() below.
        """
        x = self.question_generator(x, output_type={
            "question_1": "str",
            "question_2": "str",
            "question_3": "str",
        })
        x = self.answer_generator(x)
        return x

def get_company_info(chunk):
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
            model_name="meta-llama/Meta-Llama-3-8B-Instruct", max_new_tokens=150
        )

    def preprocess(self, obj: PromptObject):
        obj.prompt = self.make_prompt(obj)
        logger.info(f"Generating question for {obj.data['ticker']}, {obj.data['q']}")

    def postprocess(self, obj: PromptObject):
        response = obj.response
        questions = [
            response["question_1"],
            response["question_2"],
            response["question_3"],
        ]
        for question in questions:
            ans = PromptObject(prompt=question, data=obj.data.copy())
            yield ans


    def make_prompt(self, obj):
        """This is a helper function used by self.preprocess()"""
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
    def __init__(self):
        super(AnswerGenerator, self).__init__(
            model_name="meta-llama/Meta-Llama-3-8B-Instruct", max_new_tokens=150
        )

    def postprocess(self, obj: PromptObject):
        logger.info(f"Generated answer for {obj}")

    def preprocess(self, obj: PromptObject):
        obj.data["question"] = obj.prompt
        obj.prompt = self.make_prompt(obj)

    def make_prompt(self, obj: PromptObject):
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
    

async def load_earnings_calls():
    path = "/app/lamini-earnings-sdk/data/test_set_transcripts.jsonl"

    with jsonlines.open(path) as reader:
        # Here we only read the 1st line from the .jsonl file.
        for line in itertools.islice(reader, 1):
            logger.info(f"Loaded earnings call for {line['ticker']}")
            yield PromptObject(prompt="", data=line)

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
                "answer": answer.response["output"],
            }
            writer.write(answer)
            pbar.update()


async def run_pipeline():
    earnings_calls = load_earnings_calls()
    answers = QuestionAnswerPipeline().call(earnings_calls)
    await save_answers(answers)


asyncio.run(run_pipeline())
