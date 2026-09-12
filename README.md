[README.md](https://github.com/user-attachments/files/32140529/README.md)
# ChatSense AI — Search a Group Chat Properly

A small AI retrieval system for searching a messy synthetic group chat by **meaning**, not just exact words.

## Assignment fit
- 4,000 synthetic messages
- 8 participants
- 6 months
- Hinglish/code-mixed messages and typos
- 40 evaluation queries
- 8 hard queries designed with zero-word overlap
- reports overall accuracy, hard-8 accuracy and the gap
- public web UI with live queries

## Architecture
Query → multilingual sentence embedding → cosine similarity → person/time metadata reranking → top messages.

The project deliberately combines semantic retrieval with lightweight metadata-aware reranking because meaning, person and time queries have different shapes.

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

The first run downloads the multilingual Sentence Transformer model.

## Demo queries
Try:
- `When did we decide on the trip?`
- `What did Priya say about the project demo?`
- `What was the closing day for the internship application?`
- `Which day did the group finalize the hill-station trip?`  # hard
- `Where and when was the evening meal fixed?`              # hard

## Data
All chat data is synthetic and safe to publish. No real private chat is included.

## Evaluation
`data/queries.csv` contains 40 queries with the expected message ID. The Streamlit evaluation button computes:
- Overall top-1 accuracy
- Hard-8 top-1 accuracy
- Accuracy gap

## Notes
This is a compact candidate-project implementation. For a production system, add persistent vector indexing, stronger temporal parsing, conversation-window reconstruction, and a learned reranker.
