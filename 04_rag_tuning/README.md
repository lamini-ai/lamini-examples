<div align="center">
<img src="https://avatars.githubusercontent.com/u/130713213?s=200&v=4" width="110"><img src="https://huggingface.co/lamini/instruct-peft-tuned-12b/resolve/main/Lamini_logo.png?max-height=110" height="110">
</div>

# RAG Tuning

RAG inserts relevant information into the prompt by loading it from a vector database.

Run a RAG spot check:

```bash
./scripts/rag_tuning.sh
```

The RAG index will be stored in `04_rag_tuning/rag_model`.
View the RAG results at [data/results/rag_spot_check_results.jsonl](data/results/rag_spot_check_results.jsonl).

# Tune it
Similar to prompt tuning, you can tune the RAG parameters and the surrounding prompt:
- `k`: number of nearest neighbors
- `n`: number of most diverse results
    - Set to be the same as `k` to return all `k` nearest neighbors
- `batch_size` (needs to be changed before generating the index): length of each chunk returned
    - Smaller chunks tend to provide more accurate results but can increase computational overhead, larger chunks may improve efficiency but reduce accuracy.

https://github.com/lamini-ai/lamini-examples/blob/d01af0bcd91d135098f4e099f82b24b44f52d414/04_rag_tuning/lamini_rag/lamini_rag_model_stage.py#L38-L51

https://github.com/lamini-ai/lamini-examples/blob/6dc1564847e293e98182f0875e1c82bb996be20e/04_rag_tuning/lamini_rag/earnings_call_loader.py#L5-L11

# How does it work?

## You need an embedding model

An embedding model converts text into a vector embedding (a list of floating point numbers). The floating point numbers are coordinates in a vector space. In a good vector space, similar concepts will be nearby. E.g. "King" will be close to "Queen" in the space.

Every LLM is an embedding model! Here is a list of common embedding models.  https://huggingface.co/spaces/mteb/leaderboard

![image](https://github.com/lamini-ai/lamini-earnings-sdk/assets/3401278/5628406d-bd44-48f6-b5b5-4446039f5fe6)

## Embed your data

1. Convert your data into chunks.
2. Then run it through an embedding model. (Note that this is expensive because it calls an LLM)
3. Store the embedding vectors in an index (e.g. a list)
4. Compute the embedding of your query.
5. Look up most relevant matches in the index.
6. Insert them into the prompt.

![image](https://github.com/lamini-ai/lamini-earnings-sdk/assets/3401278/3ffc4f2a-e96b-4949-b7c0-ee86967d36bf)

## Optimize it

Consider more advanced optimizations described here: https://docs.google.com/presentation/d/118e4WWR4eWViJ_dTzQ5V3wwa_Eh95e5TQVklQz8hR1A/edit?usp=sharing
