# Retrieval Augmented Generation (RAG)

## Introduction

Suppose you're interested in knowing today's Bitcoin price, but the model responds:
```
I'm sorry, but my training only includes information up to Jan 2022...
```

What if you want to know the most successful product your company has launched, but the model replies:
```
I apologize, but I don't have access to real-time data or specific information about your company.
```

Language models may lack recent data and do not have access to your
private data, resulting in potentially uninformative or inaccurate replies.
This is where retrieval augmented generation (RAG) steps in, efficiently allowing users to
incorporate their internal knowledge base and/or real time data
for more accurate responses without modifying the
underlying model itself :smiley: :thumbsup:.

## High Level Overview
Lamini's `RetrievalAugmentedRunner` allows you to run RAG with just a few lines of code,
like below.
In the upcoming sections, we will provide a detailed  explanation of the RAG steps, delve into the code, and provide guidance on configuring RAG.

```python
from retrieval_augmented_runner import RetrievalAugmentedRunner

llm = RetrievalAugmentedRunner(chunk_size=512, step_size=256)
llm.load_data("data")
llm.train()
prompt = "Have we invested in any generative AI companies in 2023?"
response = llm.call(prompt)
print(response)
```
### How RAG works:
1. :books: :mag: Retrieval - Scans the knowledge base to retrieve info relevant to the user's prompt. Ex:
   - User prompt: `"Have we invested in any generative AI companies in the past year?"`
   - Scan the the company's internal documents. Retrieve information relevant to the prompt, such as company names, funding amounts, equity stakes, investments dates, and key personnel involved.
2. ➕ Augmentation - Augment the prompt with the data retrieved from step 1 like below:
   - ```
     1/1/2023, we invested in company A...
     2/1/2023, we invested in company B...
     ...

     Have we invested in any generative AI companies in 2023?
     ```
3. ✨ Generation - Generate a well-informed response for the prompt from step 2. Ex:
   - ```
     Yes, in 2023, we invested in A and B.
     ```

## Step 0: Prepare Knowledge Data

RAG requires the user to provide knowledge data which can be retrieved to augment the prompt.
Lamini requires the user to input a directory path where all files are readable as text (e.g. txt, csv).  The user can optionally specify a list of file patterns to exclude.
If any file within the specified directory cannot be read as text and is not explicity excluded, then the directory loader will fail.

## Step 1: Retrieval

### Step 1.1: Data to Chunks

To facilitate efficient processing in later stages, the initial step is to load files from the knowledge directory and segment the data into substrings.  The code below loads the files and
then breaks the file contents into chunks based the optional arguments `chunk_size` and `step_size`.

```python
llm = RetrievalAugmentedRunner(chunk_size=512, step_size=512)
llm.load_data("path/to/knowledge_directory")
```

* `chunk_size`
  - Number of characters in each chunk.  All chunks will have the same ssize, with the exception of the chunks at the end, which may be shorter.
  - Smaller chunks tend to provide more accurate results but can increase computationlly overhead.
  - Larger chunks may improve efficiency but reduce accuracy.
  - Default to 512.
* `step_size`
  - Interval at which each chunk is obtained.
  - Ex. if `step_size` = 5, then we will extract chunks from indices 0, 4, 9, 14, 19, ..., and each chunk will have length `chunk_size`.
  - Default to 128.
  - `step_size` should be less than or equal to `chunk_size`.

You can optionally specify a list of file patterns to ignore with `load_data`.
For example, the code below ignores files that end in `*.bin` and `*.exe`.
```python
llm.load_data("path/to/knowledge_directory", exclude_files=["*.bin", "*.exe"])
```

The code to load the files is very straightforward. Simply load all the files in the
directory recursively as text into a list of strings, but ignore files that [fnmatches](https://docs.python.org/3/library/fnmatch.html) `exclude_files`.
```python
    def load(self):
        # load all of the files in the directory recursively as text into a list of strings
        # return the list of strings
        for root, dirs, files in os.walk(self.directory):
            for file in files:
                exclude = False
                for pattern in self.exclude_files:
                    if fnmatch.fnmatch(file, pattern):
                        exclude = True
                        break
                if exclude:
                    continue
                with open(os.path.join(root, file), 'r') as f:
                    yield f.read()
```

The code to divide the text into chunks is also straightforward.
`DefaultChunker get_chunks` generates text chunks of a specified size, and `DirectoryLoader get_chunk_batches` uses `DefaultChunker get_chunks` to yield batches of these chunks with a specified batch size. The last batch may have shorter chunks.

```python
class DefaultChunker:
    def get_chunks(self, data):
        # return a list of strings, each a substring of the text with length self.chunk_size
        # the last element of the list may be shorter than self.chunk_size
        for text in data:
            for i in range(0, len(text), self.step_size):
                max_size = min(self.chunk_size, len(text) - i)
                yield text[i:i+max_size]

class DirectoryLoader:
    def get_chunk_batches(self):
        # A generator that yields batches of chunks
        # Each batch is a list of strings, each a substring of the text with length self.chunk_size
        # the last element of the list may be shorter than self.chunk_size
        chunks = []
        for chunk in self.get_chunks():
            chunks.append(chunk)
            if len(chunks) == self.batch_size:
                yield chunks
                chunks = []

        if len(chunks) > 0:
            yield chunks
```

#### Experiment with Chunks

You might need to experiment with adjusting these parameters to achieve optimal results.
Consider the knowledge text below:
```
"Our firm invested in 10 AI startups in 2023."
```
For simplicity, let `chunk_size` = `step_size` = 20.
In other words, for each index in [0, 20, 40], extract a substring of length 20.
This is what the output would look like:
```
["Our firm invested in",
 " 10 AI startups in 2",
 "023."]
```

Now consider `chunk_size = 20`, `step_size = 10`, meaning
for each index in [0, 10, 20, 30, 40], extract a substring of length 20.
Notice this will result in overlaps at the boundaries of adjacent chunks, as shown below.
These overlaps will give each chunk some context from its neighbors and improve result quality during later steps. Though sometimes a larger `step_size` might lead to the response containing additional content that shouldn't be included.

```
["Our firm invested in",
 "nvested in 10 AI sta",
 " 10 AI startups in 2",
 "rtups in 2023.",
 "023"]
```

You may also want to implement your own loader or chunking logic, e.g. if you are loading
data from a database instead of a set of files.  We have found that there are many different
ways of storing, loading, and transforming data.  This data loading logic usually requires
some new data wrangling.

### Step 1.2: Chunk Embeddings --> Search Index

Now that we have the chunks, we must capture the semantic information
and context of the chunks as numerical vectors known as embeddings.  A large language model
is used to convert the chunk into an embedding.  Some popular embedding LLMs are listed on
the (massive text embedding benchmark leaderboard)[https://huggingface.co/spaces/mteb/leaderboard].

We then use the embeddings to build an index, a data structure that is crucial for
efficient data retrieval in large datasets.  A simple index could just be implemented as
a list of embedding vectors.  To search through the index, a vector dot product between
the query embedding and each of the embedding vectors from the list could be used to
determine the distance in the embedding space.  An optimized library like FAISS can improve
upon this simple index by compressing it.

In Lamini, `llm.train()` performs all tasks above and saves the index to the local machine.

> For those interested, Lamini builds an [faiss.IndexFlatL2](https://github.com/facebookresearch/faiss) index, a
simple and fast index for similarity search based on Euclidean distance.

Below is the code that builds the index of the embeddings.

`build_index` initializes a Faiss index.  For each data batch, it generates the embeddings and
add the embeddings to the index. The loop also uses `tqdm` display a progress bar.
In addition, it keeps track of the data batches processed in `self.splits`.

In `get_embeddings`, `ebd.generate` invokes Lamini's embedding endpoint to generate the embeddings
for the text.

TODO: why take embedding[0]
TODO: why convert to np.array?

```python
    def build_index(self):
        self.splits = []
        self.index = None

        # load a batch of splits from a generator 
        for split_batch in tqdm(self.loader):
            embeddings = self.get_embeddings(split_batch)

            if self.index is None:
                # initialize the index
                logger.info(f"Creating index with dimension {len(embeddings[0])}")
                self.index = faiss.IndexFlatL2(len(embeddings[0]))

            # add the embeddings to the index
            self.index.add(embeddings)
            # save the splits
            self.splits.extend(split_batch)

    def get_embeddings(self, examples):
        ebd = Embedding(config=self.config)
        embeddings = ebd.generate(examples)
        embedding_list = [embedding[0] for embedding in embeddings]

        return np.array(embedding_list)
```

### Step 1.3: Retrieve Relevant Information from Embeddings

Using [FAISS](https://github.com/facebookresearch/faiss),
Lamini performs a similarity search using embeddings of the question
against all chunk embeddings, with the help of the index.
This produces a list of chunk IDs ranked by their similarity scores.

Lamini's `llm.train()` also executes this step.

By default, the search returns the top 5 IDs.  You can override this
default value by specifying `k` in the `RetrievalAugmentedRunner` config.

```python
llm = RetrievalAugmentedRunner(
      chunk_size=512,
      step_size=512,
      k=5,
)
```

Similar to adjusting `chunk_size` and `step_size`, you may need to experiment with modifying `k` to attain optimal results.

This is the query code:

```python
    def query(self, query, k=5):
        embedding = self.get_embeddings([query])[0]

        embedding_array = np.array([embedding])

        # get the k nearest neighbors
        distances, indices = self.index.search(embedding_array, k)

        return [self.splits[i] for i in indices[0]]
```

## Step 2: Augmentation

This step is simple, we prepend the relevant chunks to the original prompt.
For example:

Original prompt
```
List the worst rated projects that my company launched in 2023.
```

Augmented prompt
```
Pied Piper Compression, released 3/2023, lead by Richard H, received 1 star.
Not HotDog, released 1/2023, lead by Jian Y, received 3 stars.
New Internet, released 2/2023, lead by Richard H, received 5 stars.
Hooli Mobile Devices, released 12/2023, lead by Gavin B, received 1 star.
Nucleus, released 8/2023, lead by Nelson B, received 2 stars.

List the worst rated projects that my company launched in 2023.
```

The code below takes `prompt`, the original prompt as import and creates the augmented prompt.
```
llm.call(prompt)
```

This is the code that builds the new prompt.

```python
    def _build_prompt(self, question):
        most_similar = self.index.query(question, k=self.k)

        prompt = "\n".join(reversed(most_similar)) + "\n\n" + question

        return prompt
```

## Step 3: Generation

The final step of RAG is also very straightforward.
Execute the Runner with the new prompt.
`llm.call(prompt)` also runs this step and returns the response.

The response for the augmented prompt in the previous step may look like
```
Based on the ratings, the worst projects launched by your company in 2023 are:

Pied Piper Compression (Richard H) - 1 star
Hooli Mobile Devices (Gavin B) - 1 star
```

Awesome! :tada:

## Try Out RAG!

You can find a RAG example to experiment with [here](https://github.com/lamini-ai/sdk/tree/main/rag).
- `data` is a knowledge directory with our fictional company's
recent investment data.
- `rag.py` is a simple program that uses RAG with the investment data to answer
`"Have we invested in any generative AI companies in 2023?"`

Please follow the [installation instructions](https://github.com/lamini-ai/sdk?tab=readme-ov-file#installation-instructions) if you haven't already.

You can run `rag.py` with
```
python3 rag.py
```

Experiment with your own prompts or modify the parameters for
`RetrievalAugmentedRunner`.
Notice that updating `step_size` from 256 to the default 512 results in an incorrect response. It includes an additional investment in AquaTech Dynamics, which is not an AI company.
