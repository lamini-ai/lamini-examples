from typing import Dict

class EarningsExample:
    """ 
    Class used to encapsulate the test data used in this example.
        -- /app/lamini-earnings-sdk/data/golden_test_set.jsonl
    This class contains a few methods for handling and building 
    testing prompts used after the RAG index has been built (or loaded).

    This is separate from the DataLoader as that is expected to be used
    for RAG index building, this class is a handler to help with building
    queries to be used in the model evaluation stage.

    While having a class like this is helpful to ensure data encapsulation
    with functions acting on such data. It is not required when using a pipeline.

    Parameters
    ----------
    example: Dict[str, str]
        Example dictionary from single line in a jsonl file
         -- /app/lamini-earnings-sdk/data/golden_test_set.jsonl
    """

    def __init__(self, example: Dict[str, str]):
        self.example = example

    def get_prompt(self):
        """ prompt getter function

        Returns
        -------
        str
            Return string from the make_prompt function
        """
        return self.make_prompt()

    def get_query(self):
        """ Conduct the query construction using the relevant 
        info to inject into the query. The query will have
        the question within the example appended to the end of
        the query.

        Returns
        -------
        prompt: str
            Formatted query with relevant information and question
        """

        prompt = self.get_company_info()
        prompt += self.example["question"]

        return prompt

    def make_prompt(self):
        """ Construct a prompt using a template and inject the
        specific example information and question into the prompt.

        Returns
        -------
        prompt: str
            Formatted query with relevant information and question
        """
        prompt = "You are an expert analyst from Goldman Sachs with 15 years of experience."
        prompt += " Consider the following company: \n"
        prompt += "==========================\n"
        prompt += self.get_company_info()
        prompt += "==========================\n"
        prompt += "Answer the following question: \n"
        prompt += self.example["question"]
        return prompt


    def get_company_info(self):
        """ Construct a string using the company information
        within self.example

        Returns
        -------
        prompt: str
            Formatted query with relevant information and question
        """
        prompt = f"Date of the call: {self.example['date']}\n"
        prompt += f"Ticker: {self.example['ticker']}\n"
        prompt += f"Quarter: {self.example['q']}\n"

        return prompt