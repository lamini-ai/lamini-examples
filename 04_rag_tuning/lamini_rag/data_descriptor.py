from typing import Dict

class DatasetDescriptor:
    """ 
    A simple helper class to provide the structured output from
    generated responses in a pipeline
    """

    def get_output_type(self) -> Dict[str, str]:
        """Returns the structured output for the pipeline
        """
        return {"model_answer": "str"}