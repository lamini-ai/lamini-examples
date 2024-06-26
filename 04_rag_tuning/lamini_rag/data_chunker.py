from typing import Tuple

class BaseDataChunker:
    """ 
    Base Chunker class to handle chunking of data. This
    class is intended as a baseline to building custom Chunkers
    for use in Pipelines reading stored data.

    For a customization example of data chunking, look at the 
    EarningsCallChunker below. Within get_chunks, customize the
    prompt that is built around each data chunk. 

    A DataChunker class is expected to be used within a DataLoader
    class, where the chunker is called to chunk the loaded data. 
    Please ensure that the loader_keys used wtihin the DataLoader
    are matched with the keys extracted within get_chunks. See the
    loader_keys of 'text' and 'ticker' within the 
    EarningsCallChunker.get_chunks function example below. The load
    function within a DataLoader will be validating the data is present
    before calling the get_chunks data, so no need for validation at this
    level.

    Parameters
    ----------
    chunk_size: int
        Maximum size limit for the text chunking within get_chunks
    
    step_size: int
        Step size for the tranversing window across the text within
        get_chunks
        
    """
    def __init__(
            self, 
            chunk_size: int = 2048, 
            step_size: int = 512,
            ):
        self.chunk_size = chunk_size
        self.step_size = step_size

    def get_chunks(self):
        """Base get_chunks function used to enforce definition of
        this functions in children classes
        """
        pass


class EarningsCallChunker(BaseDataChunker):
    """ 
    Custom Chunker class to handle data chunking of the Earnings Call
    example data. The data provided to get_chunks is expected to be
    a Tuple[str, str] holding the text and ticker information in that
    order.
    """

    def get_chunks(self, data: Tuple[str, str]) -> str:
        """ A generator that yields return a list of strings, each a 
        substring of the text with length self.chunk_size the last 
        element of the list may be shorter than self.chunk_size

        This function is hard coded to work with the text and ticket
        earnings call example, this can be seen in the unpacking of the
        data parameter within the for loop below. 

        To customize this 

        Parameters
        ----------
        data: Tuple[str, str]
            Provided text and ticker string data in that order
        
        Yields
        -------
        prompt: str
            Prompt built with custom wrapper text around the relevant
            information from the provided data. Chunking is done with
            respect to the provided chunk and step sizes at instantiation
        """

        for text, ticker in data:
            for i in range(0, len(text), self.step_size):
                max_size = min(self.chunk_size, len(text) - i)

                prompt = "====================\n"
                prompt += (
                    "This is a section from an earnings call about company with ticker: "
                    + ticker
                    + "\n"
                )
                prompt += text[i : i + max_size]
                prompt += "\n"
                prompt += "====================\n"

                yield prompt

                if i + max_size == len(text):
                    break