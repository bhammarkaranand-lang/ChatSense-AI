import re
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

class ChatSearch:
    def __init__(self, csv_path="data/chat.csv"):
        self.df = pd.read_csv(csv_path)
        self.df["text"] = (
            self.df["sender"].astype(str) + " " +
            self.df["message"].astype(str)
        )
        self.model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
        self.embeddings = self.model.encode(
            self.df["text"].tolist(),
            normalize_embeddings=True,
            show_progress_bar=False
        )

    def _filters(self, query):
        q = query.lower()
        sender = next((p for p in self.df.sender.unique() if p.lower() in q), None)
        month = None
        months = {"january":1,"february":2,"march":3,"april":4,"may":5,"june":6}
        for name,num in months.items():
            if name in q:
                month=num
        return sender, month

    def search(self, query, k=5):
        qv = self.model.encode([query], normalize_embeddings=True)[0]
        scores = self.embeddings @ qv
        sender, month = self._filters(query)

        # Metadata-aware reranking for person/time-shaped queries.
        adjusted = scores.copy()
        if sender:
            adjusted += np.where(self.df.sender.str.lower().eq(sender.lower()), 0.15, 0)
        if month:
            adjusted += np.where(pd.to_datetime(self.df.date).dt.month.eq(month), 0.08, 0)

        idx = np.argsort(-adjusted)[:k]
        out = self.df.iloc[idx][["id","date","time","sender","message"]].copy()
        out["score"] = adjusted[idx]
        return out
