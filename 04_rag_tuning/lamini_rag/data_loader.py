from typing import List, Tuple
import jsonlines


class DataLoader:
    """ 
    Loader class to handle data loading example data. This
    class is a iterator containing a workflow of loading in 
    the jsonl data, chunk and batch the loaded data, then
    format and yield the relevant data. This is a support
    class for LaminiIndex to build new RAG indices.

    Parameters
    ----------
    path: str
        Path to the jsonl formatted data to chuck and yield
    
    chunker: Chunker
        Class used to format chunks into prompts. Explore
        the Chunker documentation to understand how prompts
        are formatted.
        It is a best practice to build the DataChunker in conjunction
        with this DataLoader
        
    loader_keys: List[str]
        This list holds the keys for the load function to extract
        the specified keys from the jsonl file being loaded
        
    batch_size: int
        Size of batching

    limit: int
        Upper limit of lines to read from jsonl path

    Raises
    ------
    ValueError:
        Raised when loader_keys is not provided or only an empty list is provided
        on instantiation as this is a required parameter
    
    """

    def __init__(
            self, 
            path: str, 
            chunker: BaseDataChunker,
            loader_keys: List[str],
            batch_size: int = 128, 
            limit: int = 100, 
        ):
        self.path = path
        self.batch_size = batch_size
        self.limit = limit
        self.chunker = chunker()
        if loader_keys:
            self.loader_keys = loader_keys
        else:
            raise ValueError("loader_keys is a required parameter for instantiating a DataLoader!")

    def get_chunks(self) -> str:
        """ Call and retrieve the chunks after loading the data within path

        Returns
        -------
        Result from self.chunker.get_chunks: str
            chunker should be yield formatted prompts as strings
        """
        return self.chunker.get_chunks(self.load())

    def get_chunk_batches(self) -> List:
        """ A generator that yields batches of chunks
        Each batch is a list of strings, each a substring of the 
        text with length self.batch_size the last element of the 
        list may be shorter than self.batch_size

        Yields
        -------
        chunks: List
            List of prompt formatted strings from the associated 
            self.chunker
        """
        # 
        chunks = []
        for chunk in self.get_chunks():
            chunks.append(chunk)
            if len(chunks) == self.batch_size:
                yield chunks
                chunks = []

        if len(chunks) > 0:
            yield chunks

    def load(self) -> Tuple:
        """ A generator that yields specific keys within 
        each line of the jsonl data from the specified
        path at the instantiation of this object

        Raises
        ------
        KeyError:
            This is raised when the provided self.loader_key 
            in instantiation are not present within the loaded
            jsonl file from self.path
        
        Yields
        -------
        Tuple[str, str]:
            A tuple of data values from the loader jsonl 
            file. The loader_keys provided within object
            instantiation are used to find what data to 
            extract.
            Keep in mind that keys need to be specific to 
            the data being loaded.
        """
        with jsonlines.open(self.path) as reader:
            for index, obj in enumerate(reader):
                try:
                    obj_data = [obj[key_] for key_ in self.loader_keys]
                except KeyError as e:
                    raise KeyError(f"Provided loader_keys were not all found within the provided jsonl path at line {index+1}: {e}")
                yield tuple(obj_data)

                if index >= self.limit:
                    break

    def __iter__(self):
        return self.get_chunk_batches()

    def __len__(self):
        return len(list(self.get_chunks()))
