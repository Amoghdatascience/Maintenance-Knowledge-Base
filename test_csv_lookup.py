from config import RAGConfig
from csv_lookup import CSVKnowledgeBase

config = RAGConfig()
kb = CSVKnowledgeBase(config)

print(kb.schema_summary())
print()
print(kb.count_rows())
print()
print(kb.count_clusters())
print()
print(kb.list_clusters())