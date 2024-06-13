import jsonlines
import random


def load_earnings_call_dataset():
    path = "/app/lamini-earnings-sdk/data/earnings_calls.jsonl"

    return EarningsCallsDataset(path)


class EarningsCallsDataset:
    def __init__(self, path):
        self.path = path

        self.length = self.get_length()

    def __len__(self):
        return self.length

    def __iter__(self):
        items = []

        with jsonlines.open(self.path) as reader:
            for obj in reader:
                items.append(obj)

        random.seed(42)
        random.shuffle(items)

        for index, item in enumerate(items):
            yield EarningsCallsExample(index, item)

    def get_length(self):
        return sum(1 for line in self)

    def get_output_type(self):
        return {
            "answer": "str",
            "value": "float",
            "units": "str",
        }


class EarningsCallsExample:
    def __init__(self, index, example):
        self.index = index
        self.example = example

    def get_id(self):
        return self.index

    def get_prompt(self):
        prompt = (
            "You are a financial analyst with extensive experience at Goldman Sachs. "
        )
        prompt += "You are reading questions that you have heard from a client about a specific earnings call. "
        prompt += "The question asks about specific numbers mentioned in the call. "
        prompt += "Format your answer as a json object with the following fields: { 'answer': str, 'value': float, 'units': str }. "
        prompt += "Limit value to 2 significant digits. "
        prompt += "For example if the answer is 22%, the value should be 22.0 and the units should be 'percent'. "
        prompt += "The question is about this earnings call:\n"
        prompt += "====================\n"
        prompt += f"Date of the call: {self.example['date']}\n"
        prompt += f"Ticker: {self.example['ticker']}\n"
        prompt += f"Quarter: {self.example['q']}\n"
        prompt += "====================\n"
        prompt += "The client asks\n"
        prompt += self.example["question"]
        return prompt

    def get_query(self):
        prompt = f"Date of the call: {self.example['date']}\n"
        prompt += f"Ticker: {self.example['ticker']}\n"
        prompt += f"Quarter: {self.example['q']}\n"
        prompt += self.example["question"]

        return prompt

    def is_exact_match(self, response):
        if "units" not in response:
            return False

        if "value" not in response:
            return False

        return (
            self.example["units"] == response["units"]
            and self.example["value"] == response["value"]
        )

    def get_question(self):
        return self.example["question"]

    def get_response(self, response):
        expected_response = f"Value: {self.example['value']} {self.example['units']}\n"
        if "answer" in response and isinstance(response, dict) and response["answer"] != "N/A":
            expected_response += f"Answer: {self.example['answer']}\n"

        return expected_response

    def get_response_json(self):
        return {
            "answer": self.example["answer"],
            "value": self.example["value"],
            "units": self.example["units"],
        }

    def get_default_response(self):
        return {
            "answer": "N/A",
            "value": "N/A",
            "units": "N/A",
        }

    def format_response(self, response):
        if "units" not in response:
            return "Unknown units\n"

        if "value" not in response:
            return "Unknown value\n"

        formatted_response = f"Value: {response['value']} {response['units']}\n"

        if response["answer"] != "N/A":
            formatted_response += f"Answer: {response['answer']}\n"

        return formatted_response

    def get_rubric(self):
        prompt = "Read this scoring rubric carefully and follow the instructions precisely:\n"
        prompt += "A score of 5 means that model's value is the same as the gold answer's id.\n"
        prompt += "A score of 4 means that the model's answer is the same or a paraphrase of the gold answer, but the value may not be an exact match. For example, the values '1 million' and '1000.0 thousand' are different ways of describing the same value\n"
        prompt += "A score of 3 means that the model's answer is similar as the gold answer's description, but the value may be wrong. Both answers may indicate that revenue is increased but the gold says 12 percent and the model say 50 million USD.\n"
        prompt += "A score of 2 means that the model's answer is not similar to the gold answer, but the answer is plausible.\n"
        prompt += "A score of 1 means that the model's answer is not similar to the gold answer, and the answer doesn't make sense.\n"

        prompt += "Assign a 5 for a correct value even if other fields are missing.\n"

        return prompt

