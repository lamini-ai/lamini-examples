from lamini.generation.base_prompt_object import PromptObject
from lamini.generation.embedding_node import EmbeddingNode
from lamini.generation.generation_pipeline import GenerationPipeline

import asyncio

from tqdm import tqdm

import faiss
import json
import os
import numpy as np

from lamini_rag.data_loader import DataLoader

import logging

logger = logging.getLogger(__name__)


class LaminiIndex:
    """ 
    This class is intended for the building and quering of a Faiss
    vector database, with the intention of being used for RAG queries 
    in a generation pipeline.
        
    Parameters
    ----------
    loader: DataLoader
        Object used to handle data ingestion for building an Index

    config: Dict[str, Any]
        Configuration parameters 
        
    """

    def __init__(self, loader: DataLoader = None, config = {}):
        self.loader = loader
        self.config = config

    @staticmethod
    def load_index(
        path: str,
        faiss_file_name: str = "index.faiss",
        splits_file_name: str = "splits.json"
        ):
        """Factory function for constructing a LaminiIndex object
        from existing faiss and splits files from the provided
        path. 

        Parameters
        ----------
        path: str
            Directory path to find the faiss and splits files

        faiss_file_name: str = "index.faiss"
            Faiss file name

        splits_file_name: str = "splits.json"
            Splits file name

        Returns
        -------
        lamini_index: LaminiIndex
            Instantiated LaminiIndex object from provided data
            files

        """
        lamini_index = LaminiIndex()

        faiss_path = os.path.join(path, faiss_file_name)
        splits_path = os.path.join(path, splits_file_name)

        # Load the index from a file
        lamini_index.index = faiss.read_index(faiss_path)

        # Load the splits from a file
        with open(splits_path, "r") as f:
            lamini_index.splits = json.load(f)

        return lamini_index

    async def build_index(self):
        """ 

        Parameters
        ----------
        
        
        Returns
        -------
        
        """
        self.splits = []

        total_batches = len(self.loader)

        logger.info(f"Building index with {total_batches} batches")

        # Initialize the index
        self.index = faiss.IndexFlatL2(self.loader.embedding_size)

        await self.build_index_async()


    async def build_index_async(self):
        """ 

        Parameters
        ----------
        
        
        Returns
        -------
        
        """
        total_chunks = len(self.loader)

        chunks = self.get_embedding_prompts(self.loader.get_chunks())

        embeddings = EmbeddingPipeline().call(chunks)

        await self.update_index(embeddings, total_chunks)


    async def get_embedding_prompts(self, chunks):
        """ 

        Parameters
        ----------
        
        
        Returns
        -------
        
        """
        for chunk in chunks:
            yield PromptObject(prompt=chunk)

    async def update_index(self, embeddings, total_chunks):
        """ 

        Parameters
        ----------
        
        
        Returns
        -------
        
        """
        pbar = tqdm(desc="Building index", unit=" chunks", total=total_chunks)

        async for embedding in embeddings:
            self.index.add(np.array([embedding.response]))
            self.splits.append(embedding.prompt)
            pbar.update()

    def mmr_query(self, query_embedding, k=20, n=5):
        """ 

        Parameters
        ----------
        
        
        Returns
        -------
        
        """
        embedding = query_embedding

        embedding_array = np.array([embedding])

        # get the k nearest neighbors
        distances, indices = self.index.search(embedding_array, k)

        # get the n most diverse results
        most_diverse = self.most_diverse_results(embedding, indices[0], n)

        return most_diverse

    def most_diverse_results(self, query_embedding, indices, n):
        """ 

        Parameters
        ----------
        
        
        Returns
        -------
        
        """
        # get the embeddings for the indices
        embeddings = [self.index.reconstruct(int(i)) for i in indices]

        # calculate the similarity between the query and the results
        similarities = [np.dot(query_embedding, embedding) for embedding in embeddings]

        # initialize the results
        results = [indices[0]]

        # iterate through the results
        for i in range(1, n):
            # initialize the best result
            best_result = None
            best_result_similarity = 1e9

            # iterate through the remaining results
            for j in range(len(indices)):
                # skip the result if it is already in the results
                if indices[j] in results:
                    continue

                # calculate the similarity between the result and the other results
                similarity = np.mean(
                    [np.dot(embeddings[j], embeddings[k]) for k in range(len(results))]
                )

                # update the best result
                if similarity < best_result_similarity:
                    best_result = indices[j]
                    best_result_similarity = similarity

            # add the best result to the results
            results.append(best_result)

        return [self.splits[i] for i in results]

    def save_index(self, path):
        """ 

        Parameters
        ----------
        
        
        Returns
        -------
        
        """
        faiss_path = os.path.join(path, "index.faiss")
        splits_path = os.path.join(path, "splits.json")

        logger.debug("Saving index to %s", faiss_path)
        logger.debug("Saving splits to %s", splits_path)

        logger.debug("Index size: %d", self.index.ntotal)

        # Save the index to a file
        faiss.write_index(self.index, faiss_path)

        # Save the splits to a file
        with open(splits_path, "w") as f:
            json.dump(self.splits, f)


class EmbeddingPipeline(GenerationPipeline):
    """ 
    This class is  
        
    Parameters
    ----------
    None
        
    """
    def __init__(self):
        super().__init__()

        self.embedding_generator = EmbeddingGenerator()

    def forward(self, x):
        """ 

        Parameters
        ----------
        
        
        Returns
        -------
        
        """
        x = self.embedding_generator(
            x, model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        return x


class EmbeddingGenerator(EmbeddingNode):
    """ 
    This class is  
        
    Parameters
    ----------
    None
        
    """
    
    def __init__(self):
        super().__init__(model_name="meta-llama/Llama-2-7b-chat-hf", max_tokens=5)
