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
prompt = "Have we invested in any generative AI companies in the past year?")
response = llm(prompt)
```
### How RAG works:
1. :books: :mag: Retrieval - Scans the knowledge base to retrieve info relevant to the user's prompt. Ex: 
   - User prompt: `"Have we invested in any generative AI companies in the past year?"`
   - Scan the the company's internal documents. Retrieve information relevant to the prompt, such as company names, funding amounts, equity stakes, investments dates, and key personnel involved.
2. ➕ Augmentation - Augment the prompt with the data retrieved from step 1. Ex: 
   - ```
     In the past year, we invested in two generative AI companies.
     Investment to Super Piped Piper in Palo Alto was led by Russe H. from the series A
     team and concluded on Mar 2, 2023, for $1,000,000 and 10% equity. Super Piped Piper
     focuses on ensuring responsible deployment of generative AI models.
     Details can be found in https://my_company.com/private_docs/piped_piper
     Investment to SeeFood in Palo Alto was led by Erlich B. from the seed round team
     and concluded on Oct 1, 2023, for $10,000,000 and 25% equity. SeeFood uses AI to
     create octupus cooking videos that you can see using Oculus headsets.
     Details can be found on https://my_company.com/private_docs/see_food

     Using the information above answer the following: Have we invested in any generative AI companies in the past year?
     ```
3. ✨ Generation - Generate a well-informed response for the prompt from step 2. Ex: 
   - ```
     Yes, in the past year, we invested in two generative AI companies: Super Piped Piper and SeeFood.
     ```

## Step 1: Retrieval

### Step 1.1: Data to Chunks

RAG requires the user to provide data which can be retrieved to augment the prompt.
Lamini expects this to be a directory where all files can be read as text (e.g. txt, csv).
You can load the knowledge with

```python
llm = RetrievalAugmentedRunner()
llm.load_data("~/path/to/knowledge_directory")
```

Lamini then recursively reads all files in the directory and chunks the data
into substrings, which will allow
efficient processing during ater stages.

Our `DirectoryLoader` breaks the text into chunks based on these parameters:
1. `batch_size`
   - Default to 512.
   - Each loader iteration will yield a chunk list of length `batch_size`.
2. `chunker`
   - An object that can chunk the text to a list of substrings.
   - Default to Lamini's `DefaultChunker`.

The `DefaultChunker` will fail to load the data if the input directory contains files that cannot be read as text.
The chunker depends on the paramters below and creates substrings of length `chunk_size`, with the exception of the end substring which may be shorter.

1. `chunk_size`
   - Number of characters in each chunk.
   - Smaller chunks tend to provide more accurate results but can increase computationlly overhead.
   - Larger chunks may improve efficiency but reduce accuracy.
   - Default to 512.
2. `step_size`
   - Interval at which each chunk is obtained.
   - Ex. if `step_size` = 5, then we will extract chunks from indices 0, 4, 9, 14, 19, ..., and each chunk will have length `chunk_size`.
   - Default to 128.
   - `step_size` should be less than or equal to `chunk_size`.

Consider this text:
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

You might need to experiment with adjusting these parameters to achieve optimal results.
You can configure these parameters with an optional `config` to `RetrievalAugmentRunner`, as shown below:

```python
llm = RetrievalAugmentedRunner(
   config={
      chunk_size=512,
      step_size=512,
   }
)
```


TODO: double check the config works, looks like it should

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
