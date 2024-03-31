import asyncio
import sys

sys.path.append("../03_RAG")
import csv
import logging

import jsonlines
from directory_loader import DefaultChunker, DirectoryLoader
from lamini.generation.base_prompt_object import PromptObject
from lamini.generation.generation_node import GenerationNode
from lamini.generation.generation_pipeline import GenerationPipeline
from tqdm import tqdm

logger = logging.getLogger(__name__)

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)


class QuestionAnswerPipeline(GenerationPipeline):
    def __init__(self):
        super(QuestionAnswerPipeline, self).__init__()

        self.question_generator = QuestionGenerator(
            "mistralai/Mistral-7B-Instruct-v0.1", max_new_tokens=200
        )
        self.asnwer_generator = AnswerGenerator(
            "mistralai/Mistral-7B-Instruct-v0.1", max_new_tokens=100
        )

    def forward(self, x):
        x = self.question_generator(
            x,
            output_type={
                "question_1": "string",
                "question_2": "string",
                "question_3": "string",
            },
        )
        x = self.asnwer_generator(x)
        return x


class QuestionGenerator(GenerationNode):
    def preprocess(self, obj: PromptObject):
        prompt = (
            "<s>[INST] You are an expert investment analyst working at BigMoney Ventures. "
            + "'"
            + obj.data["text"]
            + "'\nThe preceding single-quoted text is an excerpt describing various investments made by BigMoney Ventures. Generate three diverse questions about the investments.  Only generate questions that can be answered using information from the preceding single-quoted text.  Do not ask questions that require additional information outside of the preceding single-quoted text. [/INST]"
        )
        obj.prompt = prompt

    def postprocess(self, result: PromptObject):
        response = result.response
        questions = [
            response["question_1"],
            response["question_2"],
            response["question_3"],
        ]
        for question in questions:
            data = result.data.copy()
            data["question"] = question
            ans = PromptObject(prompt="", data=data)
            yield ans


class AnswerGenerator(GenerationNode):
    def preprocess(self, obj: PromptObject):
        prompt = f"""<s>[INST] You are an expert in the field of investments. '{obj.data["text"]}' The preceding single-quoted text is an excerpt describing various investment made by BigMoney Ventures.  Answer the following question using information from the single-quoted text.  If you cannot answer the question using only the single-quoted text, respond only with the statement: \"I don't know.\" {obj.data["question"]}[/INST]"""
        obj.prompt = prompt


def get_prompt_generator(loader):
    for example in loader:
        for chunk in example:
            yield PromptObject("", data={"text": chunk})


async def save_predictions(results):
    # Save the questions, answers, and data in a csv file (logging)
    csv_file = open("qa_data/generated_data.csv", "w")
    writer = csv.writer(csv_file)
    writer.writerow(["Data", "Questions", "Answers"])

    # And finally, save the format that will actually be submitted to the model for finetuning
    training_file = jsonlines.open("qa_data/generated_data_finetuning.jsonl", "w")
    pbar = tqdm(desc="Saving predictions", unit=" predictions")
    async for obj in results:
        writer.writerow(
            [obj.data["text"], obj.data["question"], obj.response["output"]]
        )
        training_data = {
            "data": obj.data["text"],
            "question": obj.data["question"],
            "answer": obj.response["output"],
        }

        training_file.write(training_data)
        pbar.update()

    csv_file.close()
    training_file.close()


async def main():
    # Load the data with the DirectoryLoader
    loader = DirectoryLoader(
        "../03_RAG/data",  # path to data directory
        batch_size=512,
        chunker=DefaultChunker(chunk_size=512, step_size=512),
    )

    # Construct a stream of Prompt Objects
    prompts = get_prompt_generator(loader)

    pipeline = QuestionAnswerPipeline()
    results = pipeline.call(prompts)

    await save_predictions(results)


asyncio.run(main())
