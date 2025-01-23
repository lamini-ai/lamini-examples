from typing import Generator, Dict, Any

import jsonlines
import random


class EarningsCallsExample:
    """
    A class holding the id, prompt, and formatting of queries
    for the Earnings Call examples dataset. This class is intended
    to be used within a Dataset object to format lines from a
    jsonlines file for a Lamini.generate call on a single example.

    Parameters
    ----------
    index: int
        Index location of the example

    example: Dict[str, Any]
        Key/Value pairs of information for the jsonline example

    """

    def __init__(self, index, example) -> None:
        self.index = index
        self.example = example

    def get_id(self) -> int:
        """ Getter function for the example ID

        Parameters
        ----------
        None

        Returns
        -------
        self.index: int
            Index location of the example in a jsonlines file
        """

        return self.index

    def get_prompt(self) -> str:
        """ Getter function to build the prompt format using
        information within the self.example dictionary.

        Parameters
        ----------
        None

        Returns
        -------
        prompt: str
            Formatting prompt string containing self.example information
        """

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

    def get_query(self) -> str:
        """ Getter function to build a query using this example data

        Parameters
        ----------
        None

        Returns
        -------
        prompt: str
            Formatting prompt string containing self.example information
        """

        prompt = f"Date of the call: {self.example['date']}\n"
        prompt += f"Ticker: {self.example['ticker']}\n"
        prompt += f"Quarter: {self.example['q']}\n"
        prompt += self.example["question"]

        return prompt

    def is_exact_match(self, response: Dict[str, Any]) -> bool:
        """ Comparison of a response from Lamini.generate with the
        example data.

        Parameters
        ----------
        response: Dict[str, Any]
            Response information from Lamini.generate

        Returns
        -------
        bool
            Comparison result between example and response
        """

        if "units" not in response:
            return False

        if "value" not in response:
            return False

        return (
            self.example["units"] == response["units"]
            and self.example["value"] == response["value"]
        )

    def get_question(self) -> str:
        """ Getter function for the question within the example
        data

        Parameters
        ----------
        None

        Returns
        -------
        str
            Question string within the example data
        """

        return self.example["question"]

    def get_response(self, response: Dict[str, Any]) -> str:
        """ Getter function for the expected response for this example

        Parameters
        ----------
        response: Dict[str, Any]
            Response dictionary from Lamini.generate call using this
            example

        Returns
        -------
        expected_response: str
            Expected returned string from Lamini.generate call for this
            example
        """

        expected_response = f"Value: {self.example['value']} {self.example['units']}\n"
        if "answer" in response and isinstance(response, dict) and response["answer"] != "N/A":
            expected_response += f"Answer: {self.example['answer']}\n"

        return expected_response

    def get_response_json(self) -> Dict[str, Any]:
        """ Getter function for the expected response in json format

        Parameters
        ----------
        None

        Returns
        -------
        Dict[str, Any]
            Expected returned json response from Lamini.generate for this
            example
        """

        return {
            "answer": self.example["answer"],
            "value": self.example["value"],
            "units": self.example["units"],
        }

    def get_default_response(self) -> Dict[str, str]:
        """ Getter function for the default json response

        Parameters
        ----------
        None

        Returns
        -------
        Dict[str, str]
            Expected defautl json response from Lamini.generate
        """

        return {
            "answer": "N/A",
            "value": "N/A",
            "units": "N/A",
        }

    def format_response(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """ Response dictionary formatting

        Parameters
        ----------
        response: Dict[str, Any]
            Returned json response from Lamini.generate

        Returns
        -------
        Dict[str, str]
            Formatted 'Value' from the provided response
        """

        if "units" not in response:
            return "Unknown units\n"

        if "value" not in response:
            return "Unknown value\n"

        formatted_response = f"Value: {response['value']} {response['units']}\n"

        if response["answer"] != "N/A":
            formatted_response += f"Answer: {response['answer']}\n"

        return formatted_response

    def get_rubric(self) -> str:
        """ Rubric string construction and format

        Parameters
        ----------
        None

        Returns
        -------
        prompt: str
            Formatted rubric for queries in Lamini.generate
        """

        prompt = "Read this scoring rubric carefully and follow the instructions precisely:\n"
        prompt += "A score of 5 means that model's value is the same as the gold answer's id.\n"
        prompt += "A score of 4 means that the model's answer is the same or a paraphrase of the gold answer, but the value may not be an exact match. For example, the values '1 million' and '1000.0 thousand' are different ways of describing the same value\n"
        prompt += "A score of 3 means that the model's answer is similar as the gold answer's description, but the value may be wrong. Both answers may indicate that revenue is increased but the gold says 12 percent and the model say 50 million USD.\n"
        prompt += "A score of 2 means that the model's answer is not similar to the gold answer, but the answer is plausible.\n"
        prompt += "A score of 1 means that the model's answer is not similar to the gold answer, and the answer doesn't make sense.\n"

        prompt += "Assign a 5 for a correct value even if other fields are missing.\n"

        return prompt

class EarningsCallsDataset:
    """
    A Dataset handler for iteration as well as output type formatting for
    Lamini generate calls.

    Parameters
    ----------
    path: str
        Dataset path string

    """

    def __init__(self, path: str) -> None:
        self.path = path

        self.length = self.get_length()

    def __len__(self) -> int:
        """ Length measured of lines within the json lines

        Parameters
        ----------
        None

        Returns
        -------
        int:
            Number of lines, as definied within __init__
        """

        return self.length

    def __iter__(self) -> Generator[EarningsCallsExample, None, None]:
        """ Iteration of the provided jsonlines file

        Parameters
        ----------
        None

        Yields
        ------
        EarningsCallsExample:
            Number of lines, as definied within __init__
        """
        items = []

        with jsonlines.open(self.path) as reader:
            for obj in reader:
                items.append(obj)

        random.seed(42)
        random.shuffle(items)

        for index, item in enumerate(items):
            yield EarningsCallsExample(index, item)

    def get_length(self) -> int:
        """ Return the number of lines in the jsonlines file

        Parameters
        ----------
        None

        Yields
        ------
        int:
            number of lines within the jsonlines file
        """

        return sum(1 for line in self)

    def get_output_type(self) -> Dict[str, str]:
        """ Return the dictionary for the output format
        of a Lamini.generate call.

        Parameters
        ----------
        None

        Returns
        -------
        Dict[str, str]:
            Output format of a generate call from Lamini
        """

        return {
            "answer": "str",
            "value": "float",
            "units": "str",
        }

def load_earnings_call_dataset() -> EarningsCallsDataset:
    """ Wrap jsonlines file into an EarningsCallsDataset

    Parameters
    ----------
    None

    Returns
    -------
    EarningsCallsDataset
        Object to handle loading and output types
    """

    path = "../data/earnings_calls.jsonl"

    return EarningsCallsDataset(path)
