<div align="center">
<img src="https://avatars.githubusercontent.com/u/130713213?s=200&v=4" width="110"><img src="https://huggingface.co/lamini/instruct-peft-tuned-12b/resolve/main/Lamini_logo.png?max-height=110" height="110">
</div>

# Retrieval Augmented Generation (RAG) Tuning
Improve the quality of your LLM by adding and tuning information retrieval. RAG inserts relevant information into your LLM prompt by loading it from an embedding.

Run RAG:
```bash
cd 04_rag_tuning
python3 rag.py
```

# Tune it
Similar to prompt tuning, you can tune the RAG parameters and the surrounding prompt:
- `k`: number of nearest neighbors (chunks) to use
- `prompt`: the prompt surrounding the RAG chunks and the question
