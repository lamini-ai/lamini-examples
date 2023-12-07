# Retrieval Augmented Generation (RAG)

## Introduction

Suppose you're interested in knowing today's Bitcoin price, but the model responds.
```
I'm sorry, but my training only includes information up to Jan 2022...
```

What if you want to know the worst products your company launched in 2023, but the model replies
```
I apologize, but I don't have access to real-time data or specific information about your company.
```

Language models may lack recent data and do not have access to your
private data, resulting in potentially uninformative or inaccurate replies.
In addition, models can't learn new information without going through a computationally
expensive and time consuming retraining process to modify the model :cry:.
This is where retrieval augmented generation (RAG) steps in, efficiently allowing users to
incorporate their internal knowledge base for more accurate responses without modifying the
underlying model itself :smiley: :thumbsup:.

Here is a high level overview of the RAG process.

User Input:
1. Prompt
2. Internal knowledge base - files containing internal knowledge

RAG Steps:
1. :books: :mag: Retrieval - Scan the knowledge base to retrieve info relevant to the user prompt. Ex.
   - User prompt `"Have we invested in any generative AI companies in the past year?"`
   - Scan the user's knowledge base, which includes the company's internal documents. Retrieve information relevant to the prompt, such as company names, funding amounts, equity stakes, investments dates, and key personnel involved.
2. :heavy_plus_sign: Augmentation - Augment the prompt with the data retrieved from step 1. 
3. :magic_wand: Generation - Generate a well-informed response for the prompt from step 2. Ex.
   - ```
     Yes, in the past year, we invested in two generative AI companies.
     Investment to Super Piped Piper in Palo Alto was led by Russe H. from the series A
     team and concluded on Mar 2, 2023, for $1,000,000 and 10% equity. Super Piped Piper
     focuses on ensuring responsible deployment of generative AI models.
     Details can be found in https://my_company.com/private_docs/piped_piper
     Investment to SeeFood in Palo Alto was led by Erlich B. from the seed round team
     and concluded on Oct 1, 2023, for $10,000,000 and 25% equity. SeeFood uses AI to
     create octupus cooking videos that you can see using Oculus headsets.
     Details can be found on https://my_company.com/private_docs/see_food
     ```

Lamini's `RetrievalAugmentedRunner` allows you to run RAG with just a few lines of code,
like below.
We will provide a detailed explanation of the RAG steps and the code in the upcoming sections.

```python
from llama.retrieval.retrieval_augmented_runner import RetrievalAugmentedRunner

llm = RetrievalAugmentedRunner()
llm.load_data("path_to_knowledge_directory")
llm.train()
prompt = "Have we invested in any generative AI companies in the past year?")
response = llm(prompt)
```

## Step 0: Prepare Input

RAG requires the user to provide an internal knowledge base along with the prompt.
Lamini expects this knowledge base to be a directory where all files can be read as text.
You can load the knowledge with

```python
llm = RetrievalAugmentedRunner()
llm.load_data("path_to_knowledge_directory")
```



## Step 1: Retrieval

### Preparing Knowledge Base as Chunks

We begin by breaking the list of text files containing the internal knowledge into chunks,
which will allow efficient and scalable processing during later stages.

```python
loader = DirectoryLoader(
    parent_directory + "/data",
    batch_size=512,                                       
    chunker=DefaultChunker(chunk_size=512, step_size=512),
)

chunks = []
for chunk in tqdm(loader):
    chunks.extend(chunk) 
```

`DirectoryLoader(...)` has three arguments:
1. A directory path (required, string)
   - path to the directory with internal data, the files will be read as text.  This step will fail if the directory contains files that cannot be read as text. TODO: does text from multiple files get combined. TODO: does it have to be abs path.
2. `batch_size` (optional keyword arg)
   - default to 512.
3. `chunker` (optional keyword arg)
   - an object that can chunk the text. Default to lamini's `DefaultChunker`, which returns a list of strings, each a substring with chunk size as the length. TODO: except at the end.

TODO: what is tqdm

If you use `DefaultChunker`, then you may specify two arguments:
1. `chunk_size` (optional keyword arg)
   - Number of characters for each chunk.
   - Smaller chunks tend to provide more accurate results but can increase computationlly overhead.
   - Larger chunks may improve efficiency but reduce accuracy.
   - Default to 512.
2. `step_size` (optional keyword arg)
   - Interval at which each chunk is obtained.
   - Ex. if `step_size` = 5, then we will extract chunks from indices 0, 4, 9, 14, 19, ..., and each chunk will have length `chunk_size`.
   - Default to 128.
   - `step_size` should be less than or equal to `chunk_size`.

`DefaultChunker` returns a list of strings, each a substring of the text with length `chunk_size`.

Consider this text:
```
"Our firm invested in 10 AI startups in 2023."
```
For simplicity, let's try `chunk_size` = `step_size` = 20.
In other words, for each index in [0, 20, 40], extract a substring of length 20.
Output:
```
["Our firm invested in",
 " 10 AI startups in 2",
 "023."
```

Now consider `chunk_size` = 20, `step_size` = 10, meaning
for each index in [0, 10, 20, 30, 40], extract a substring of length 20.
Notice this will result in overlaps at the boundaries of adjacent chunks, as shown below.
These overlaps will give each chunk some context from its neighbors and improve result quality during later steps.

```
["Our firm invested in",
 "nvested in 10 AI sta",
 " 10 AI startups in 2",
 "rtups in 2023.",
 "023"
```

How to choose the right chunk and step sizes?
TODO

TODO: Smaller chunks often improve retrieval but may cause generation to suffer from a lack of surrounding context. Is this related to step size?

### Chunks --> Embeddings and Index

Now that we have the list of strings as chunks, we must capture the semantic information
and context of the chunks
as numerical vectors known as embeddings.  This enables the data to be processed effectively
by machine learning models.

TODO: add embedding example

Next, we must use the embeddings to build an index, a data structure that is crucial for
efficient data retrieval in large datasets.  An index is essentially a map, helping you
find specific information quickly, just like the index at the end of a book.
Lamini builds an `faiss.IndexFlatL2` index (TODO: link), a
simple and fast index for similarity search based on Euclidean distance.


`llm.train()` will create an index
```python
self.index = LaminiIndex(self.loader, ...)
````

The index is saved.

### Retrieve Relevant Information from Embedding Store

We perform a similarity search using embeddings of the question
again all embeddings in the embedding store.  This produces a list
of chunk IDs ranked by their similarity scores.

TODO: pre/post filtering, see https://scale.com/blog/retrieval-augmented-generation-to-enhance-llms

## Step 2 Augmentation

Append the relevant chunks to the original prompt.

The original prompt is augmented with the additional information from the knowledge base.
For example, original prompt:

```
List the worst rated projects that my company launched in 2023.
```

Augmented prompt:
```
List the worst rated projects that my company launched in 2023.
Consider these:
Pied Piper Compression, released 3/2023, lead by Richard H, received 1 star.
Not HotDog, released 1/2023, lead by Jian Y, received 3 stars.
New Internet, released 2/2023, lead by Richard H, received 5 stars.
Hooli Mobile Devices, released 12/2023, lead by Gavin B, received 1 star.
Nucleus, released 8/2023, lead by Nelson B, received 2 stars.
```

## Step 3 Generation

The final step of RAG is to execute the Runner with the new prompt.

The output will look like:
```
Based on the ratings, the worst projects launched by your company in 2023 are:

Pied Piper Compression (Richard H) - 1 star
Hooli Mobile Devices (Gavin B) - 1 star
Nucleus (Nelson B) - 2 stars
```

Awesome! :tada: Now, follow along the next topic to learn how to train models with
[instruction, output] pairs to get even better responses!

==================== IGNORE STUFF BELOW =====

add figure?

We encode each chunk into an embedding vector and use that for retrieval

Text can be chunked and vectorized externally and then indexed as vector fields in your index.
To encode that data, we need to use an embedding model. 

LaminiIndex() builds the index, creates splits
for each in split_batch, get embeddings
  set index to faiss.IndexFlatL2


=======================

step 1: load, concat
step 2: chunk fixed, show code
step 3: run embedding

TODO: runner default is mistral

TODO add figure?





