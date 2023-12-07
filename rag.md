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
This is where retrieval augmented generation (RAG) steps in, efficiently allowing users to
incorporate their internal knowledge base or real time data
for more accurate responses without modifying the
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
## Step 1. Retrieval

### Step 1.1 Internal Knowledge Base to Chunks

RAG requires the user to provide an internal knowledge base along with the prompt.
Lamini expects this knowledge base to be a directory where all files can be read as text.
You can load the knowledge with

```python
llm = RetrievalAugmentedRunner()
llm.load_data("path_to_knowledge_directory")
```

Lamini then recursively reads all files in the directory and chunks the data
into substrings, which will allow
efficient processing during ater stages.

Our `DirectoryLoader` breaks the text into chunks based on these parameters:
1. `batch_size`
   - Default to 512.
   - TODO: what does batch size do here?
2. `chunker`
   - An object that can chunk the text to a list of substrings.
   - Default to lamini's `DefaultChunker`.

If you our `DefaultChunker`, the data will fail to load if the input directory contains
files that cannot be read as text (TODO: double check).
The chunker depends on the paramters below and creates substrings of length `chunk_size`, except
possibly shorter substrings at the end.  We will show some examples.

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
For simplicity, let `chunk_size` = `step_size` = 20.
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

You might need to experiment with adjusting these parameters to achieve optimal results.
You can configure these parameters with an optional `config` to `RetrievalAugmentRunner`, like below:

```python
llm = RetrievalAugmentedRunner(
    config={'chunk_size'=512,
            'step_size'=512,
            'batch_size'=512,
           })
```


TODO: double check the config works, looks like it should

TODO: you can also specify k, what is k?

### Steps 1.2: Chunks :arrow_right: Embeddings

Now that we have the chunks, we must capture the semantic information
and context of the chunks
as numerical vectors known as embeddings.
This enables the data to be processed effectively by machine learning models.

TODO: add details of how we do embedding?

### Step 1.3: Embeddings :arrow_right: Search Index

Next, we must use the embeddings to build an index, a data structure that is crucial for
efficient data retrieval in large datasets.  An index is essentially a map, helping you
find specific information quickly, just like the index at the end of a book.
Lamini builds an `faiss.IndexFlatL2` index (TODO: link), a
simple and fast index for similarity search based on Euclidean distance.
Lamini's `llm.train()` builds the index and saves the index file to the local machine.

### Step 1.4: Retrieve Relevant Information from Embedding Store

We perform a similarity search using embeddings of the question
again all embeddings.  This produces a list
of chunk IDs ranked by their similarity scores.

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





