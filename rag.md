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
from llama import RetrievalAugmentedRunner

llm = RetrievalAugmentedRunner()
llm.load_data("path/to/knowledge_directory")
llm.train()
prompt = "Have we invested in any generative AI companies in 2023?")
response = llm(prompt)
```
### How RAG works:
1. :books: :mag: Retrieval - Scans the knowledge base to retrieve info relevant to the user's prompt. Ex: 
   - User prompt: `"Have we invested in any generative AI companies in the past year?"`
   - Scan the the company's internal documents. Retrieve information relevant to the prompt, such as company names, funding amounts, equity stakes, investments dates, and key personnel involved.
2. ➕ Augmentation - Augment the prompt with the data retrieved from step 1 like below:
   - ```
     1/1/2023, we invested in company A...
     2/1/2023, we invested in company B...
     3/1/2023, we invested in company C...     
     ...

     Using the information above answer the following: Have we invested in any generative AI companies in the past year?
     ```
3. ✨ Generation - Generate a well-informed response for the prompt from step 2. Ex: 
   - ```
     Yes, in 2023, we invested in A and B.
     ```

## Step 0: Prepare Knowledge Data

RAG requires the user to provide knowledge data which can be retrieved to augment the prompt.
Lamini requires the user to input a directory path where all files are readable as text (e.g. txt, csv). If any file within the specified directory cannot be read as text, then the directory loader will fail.

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

You might need to experiment with adjusting these parameters to achieve optimal results.
Consider the knowledge text below:
```
"Our firm invested in 10 AI startups in 2023."
```
For simplicity, let `chunk_size` and `step_size` to 20.
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
These overlaps will give each chunk some context from its neighbors and improve result quality during later steps.

```
["Our firm invested in",
 "nvested in 10 AI sta",
 " 10 AI startups in 2",
 "rtups in 2023.",
 "023"]
```

### Step 1.2: Chunk Embeddings --> Search Index

Now that we have the chunks, we must capture the semantic information
and context of the chunks
as numerical vectors known as embeddings.
This enables the data to be processed effectively by machine learning models.

We then use the embeddings to build an index, a data structure that is crucial for
efficient data retrieval in large datasets.  An index is essentially a map, helping you
find specific information quickly, just like the index at the end of a book.


In Lamini, `llm.train()` performs all tasks above and saves the index to the local machine.

> For those interested, Lamini builds an [faiss.IndexFlatL2](https://github.com/facebookresearch/faiss) index, a
simple and fast index for similarity search based on Euclidean distance.

### Step 1.3: Retrieve Relevant Information from Embeddings

Using [faiss](https://github.com/facebookresearch/faiss),
Lamini performs a similarity search using embeddings of the question
against all chunk embeddings, with the help of the index.
This produces a list of chunk IDs ranked by their similarity scores.

Lamini's `llm.train()` also executes this step.

By default, the search returns the top 5 IDs.  You can override this
default value by specifying `k` in the `RetrievalAugmentedRunner` config.

```python
llm = RetrievalAugmentedRunner(
   config={
      chunk_size=512,
      step_size=512,
      k=5,
   }
)
```

Similar to adjusting `chunk_size` and `step_size`, you may need to experiment with modifying `k` to attain optimal results.

## Step 2: Augmentation

This step is simple, we append the relevant chunks to the original prompt.
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

In (TODO: add location), line TODO
```
llm(prompt)
```
creates the augments prompt.

## Step 3: Generation

The final step of RAG is also very straightforward.
Execute the Runner with the new prompt.
`llm(prompt)` also runs this step and returns the response.

The response for the augmented prompt in the previous step may look like
```
Based on the ratings, the worst projects launched by your company in 2023 are:

Pied Piper Compression (Richard H) - 1 star
Hooli Mobile Devices (Gavin B) - 1 star
```

TODO: this output is from chat gpt, double check our model produces similiar response.

Awesome! :tada:

===========================

TODO: add code that the user can run
