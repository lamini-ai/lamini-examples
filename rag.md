# Retrieval Augmented Generation (RAG)

## Introduction

Let's say you want to the know the bitcoin price today, but the model responds
```
I'm sorry, but my training only includes information up to Jan 2022...
```

What if you want to know the list of products your company launched in 2020 that received
high ratings, but the model replies
```
I apologize, but I don't have access to real-time data or specific information about your company.
```

Language models may lack recent data and do not have access to your
private data, resulting in uninformative replies or potential inaccuracies or hallucinations in the replies.
In addition, models can't learn new information without going through a computationally
expensive and time consuming retraining process to modify the model :cry:.
This is where retrieval augmented generation (RAG) steps in, efficiently allowing users to
incorporate their knowledge base for more accurate answers without modifying the
underlying model itself :smiley: :thumbsup:.

To use RAG, the user will provide a knowledge base along with the prompt.
In a later section, we will guide you through preparing the knowledge base.
Once ready, RAG will perform the steps below:
1. :mag: Retrieval - Scan the knowledge base to retrieve info relevant to the user prompt. Ex.
   - User prompt `"Have we invested in any generative AI companies in the past year?"`
   - RAG searches the user's knowledge base, which includes the company's internal documents to retrieve information relevant to the prompt, such as the recipent companies, funding amounts, equity stakes, investments dates, and key personnel involved.
2. :books: :heavy_plus_sign: :books: Augmentation - Augment the prompt with the retrieved data from step 1.
3. :magic_wand: Generation - Generate a well-informed response for the prompt from step 2. Ex.
   - ```
     Yes, in the past year, we invested in two generative AI companies.
     Investment to Super Piped Piper in Palo Alto was led by Russe H. from the series A
     team and concluded in Mar 2, 2023, for $1,000,000 and 10% equity. Super Piped Piper
     focuses on ensuring responsible deployment of AI models.
     Details can be found in https://my_company.com/private_docs/piped_piper
     Investment to SeeFood in Palo Alto was led by Erlich B. from the seed round team
     and concluded in Oct 1, 2023, for $10,000,000 and 25% equity. SeeFood uses AI to
     create octupus cooking videos that you can see using Oculus headsets.
     Details can be found in https://my_company.com/private_docs/see_food
     ```

Once the knowledge base is ready, you can use lamini's `RetrievalAugmentedRunner`
to get the result above with just a few lines of code.

```python
import lamini

llm = RetrievalAugmentedRunner() # default model is Mistral Instruct
llm.load_data("private_knowledge_dir")
llm.train()
response = llm("Have we invested in any generative AI companies in the past year?")
```

Now let's delve into how RAG works.

## Step 1: Retrieval

### Knowledge Base Prepartion

We begin by breaking the list of text files containing the internal knowledge into chunks,
which will allow efficient and scalable processing, indexing and retrieval.

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
1. `directory_path` (required) - path to the directory with internal data, the files will be read as text.  This step will fail if the directory contains files that cannot be read as text. TODO: does text from multiple files get combined.
2. `batch_size` (optional) - default to 512.
3. `chunker` (optional) - an object that can chunk the text. The default chunker is lamini's `DefaultChunker`.

If you use `DefaultChunker`, then you can optionally specify two arguments:
1. `chunk_size` (optional)
   - Number of characters for each chunk.
   - Smaller chunks tend to provide more precise results but can increase computationlly overhead.
   - Larger chunks may improve efficiency but introduce more noise in the results.
   - Default to 512.
2. `step_size` (optional)
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

Now let `chunk_size` = 20, `step_size` = 16.
This means for each index in [0, 16, 32, 48], extract a substring of length 20.
Since `chunk_size` > `step_size`, the resulting chunks contain overlapping subsequences.
```
["Our firm invested in",
 "ested in 10 AI start",
 "AI startups in 2023.",
 "in 2023."
```

How to choose the right chunk and step sizes?
TODO

=========================

In this step, you simply need to provide a directory of text files that contains
your internal knowledge.

In this step, we will input a directory of text files and break the input text into
a list of special substrings or chunks.


Prepare a directory with your text files.
RAG will recursively load all the files as text into a list of strings.
Then chunk the data into
a list of strings, each a substring of the text with length self.chunk_size.
 chunking is the process of breaking down the input text into smaller segments or chunks. A chunk could be defined simply by its size
 

We encode each chunk into an embedding vector and use that for retrieval
If the chunk is small enough it allows for a more granular match between the user query and the content, whereas larger chunks result in additional noise in the text, reducing the accuracy of the retrieval step.

chunk by token size

Text can be chunked and vectorized externally and then indexed as vector fields in your index.

add input ex, and out chunks

To encode that data, we need to use an embedding model. 

LaminiIndex() builds the index, creates splits
for each in split_batch, get embeddings
  set index to faiss.IndexFlatL2



 In general, smaller chunks often improve retrieval but may cause generation to suffer from a lack of surrounding context.
 Chunking significantly influences the quality of the generated content.

. The findings suggested that larger chunk sizes can be beneficial, but the benefits taper off after a certain point, indicating that too much context might introduce noise.

Overlap in chunking refers to the intentional duplication of tokens at the boundaries of adjacent chunks. This ensures that when a chunk is passed to the LLM for generation, it contains some context from its neighboring chunks, enhancing the quality of the output.


## Step 2: Augmentation


## Step 3: Generation


=======================

step 1: load, concat
step 2: chunk fixed, show code
step 3: run embedding

runner default is mistral

figure


Advantages of Retrieval Augmented Generation

Scalability. Instead of having a monolithic model that tries to memorize every bit of information, RAG models can scale by simply updating or enlarging the external database.
Memory efficiency. While traditional models like GPT have limits on the amount of data they can store and recall, RAG leverages external databases — allowing it to pull in fresh, updated or detailed information when needed.
Flexibility. By changing or expanding the external knowledge source, you can adapt a RAG model for specific domains without retraining the underlying generative model.

