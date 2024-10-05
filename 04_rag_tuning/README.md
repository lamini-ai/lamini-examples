# Retrieval Augmented Generation (RAG) Tuning

Improve the quality of your LLM by adding and tuning information retrieval. RAG inserts relevant information into your LLM prompt by loading it from an embedding.

Run RAG:

```bash
cd 04_rag_tuning
python3 rag.py
```

## Tune it

Similar to prompt tuning, you can tune the RAG parameters and the surrounding prompt:

- `k`: number of nearest neighbors (chunks) to use
- `prompt`: the prompt surrounding the RAG chunks and the question
