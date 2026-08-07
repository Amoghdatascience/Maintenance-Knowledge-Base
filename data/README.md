# Cluster-Focused RAG Knowledge Base

Generated from `final_joined_output.csv`.

This folder intentionally removes generic Agentic RAG tutorial information and keeps only cluster-focused knowledge.

Files:
- `cluster_index.txt`: overview of all clusters and record counts
- `cluster_<id>.txt`: one RAG document per cluster

Recommended use:
1. Delete unrelated tutorial `.txt` files from your project's `data/` folder.
2. Keep `final_joined_output.csv` in `data/`.
3. Copy these cluster `.txt` files into `data/`.
4. Rebuild Chroma.

Commands:

```bash
cd /Users/amoghnaik/gpt_rags
rm -f data/agentic_rag_overview.txt data/langgraph_architecture.txt data/chunking_and_embeddings.txt data/free_local_models.txt data/rag_evaluation_guide.txt data/troubleshooting_rag_pipeline.txt
cp /path/to/cluster_rag_knowledge_base/*.txt data/
rm -rf chroma_db
python ingest.py
python main.py
```

Your hybrid system should work like this:
- CSV_LOOKUP: exact counts, lists, filters, records, cluster IDs
- VECTOR_RAG: explanation of cluster meaning, inclusion guidance, common terms, examples
