# Retrieval Augmented Generation (RAG)

## Introduction

Language models may lack recent data and do not have access to your
private data, resulting in potential inaccuracies or hallucinations in the
repsonses.
That's where retrieval augmented generation (RAG) steps in, allowing users to
incorporate their knowledge base for more accurate answers without modifying the
underlying model itself.

To use RAG, the user will provide a knowledge base along with a prompt.
Then, RAG will perform the steps below:
1. <u>R</u>etrieval - Scan the knowledge base to retrieve info relevant to the user prompt. Ex.
   - User prompt `"Have we invested in any generative AI companies in the past year"`?
   - RAG searches the user's knowledge base, which includes the company's internal documents and databases to retrieve information relevant to the prompt, such as the recipent companies, funding amounts, equity stakes, investments dates, key personnel involved.
2. <u>A</u>ugmentation - Augment the prompt with the retrieved data from step 1.
3. <u>G</u>eneration - Generate a well-informed response for the prompt from step 2. Ex.
   - ```
     Yes, in the past year, we have invested in two companies. Investment to Piped Piper
     in Palo Alto was led by Erlich Bachman from the series A team and concluded in
     Mar 2, 2023, for $1,000,000 and 10% equity.  Investment to SeeFood in Palo Alto was
     led by Jian Yang from the seed round team  and concluded in Oct 1, 2023, for
     $10,000,000 and 25% equity.
     ```

In this tutorial, we will show you how to use lamini's
`RetrievalAugmentedRunner` to obtain high quality results.

```python
llm = RetrievalAugmentedRunner()
llm.load_data("my_data_dir")
llm.train()
response = llm("Who won ...?")
```

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

