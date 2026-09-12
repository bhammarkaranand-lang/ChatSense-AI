import streamlit as st
import pandas as pd
from search_engine import ChatSearch

st.set_page_config(page_title="ChatSense AI", page_icon="💬", layout="wide")
st.title("💬 ChatSense AI")
st.caption("Semantic + metadata-aware search for messy group chats")

@st.cache_resource
def load_engine():
    return ChatSearch("data/chat.csv")

engine = load_engine()

st.sidebar.header("About")
st.sidebar.write("4,000 synthetic messages • 8 participants • 6 months")
st.sidebar.write("Supports semantic meaning, Hinglish/code-mixed text, person and time hints.")

query = st.text_input(
    "Search the group chat",
    placeholder="e.g. When did we decide on the trip?"
)

k = st.slider("Results", 1, 10, 5)

if query:
    results = engine.search(query, k)
    st.subheader("Relevant messages")
    for _, r in results.iterrows():
        with st.container(border=True):
            st.markdown(f"**{r['sender']}**  ·  {r['date']} {r['time']}  ·  `{r['id']}`")
            st.write(r["message"])
            st.caption(f"semantic/reranked score: {r['score']:.3f}")

st.divider()
st.subheader("Evaluation")
if st.button("Run 40-query evaluation"):
    qdf = pd.read_csv("data/queries.csv")
    hits=[]
    hard=[]
    progress=st.progress(0)
    for i,row in qdf.iterrows():
        top = engine.search(row["query"], 1)
        hit = len(top) and top.iloc[0]["id"] == row["answer_id"]
        hits.append(bool(hit))
        if row["type"] == "hard":
            hard.append(bool(hit))
        progress.progress((i+1)/len(qdf))
    overall = sum(hits)/len(hits)*100
    hard_acc = sum(hard)/len(hard)*100
    c1,c2,c3=st.columns(3)
    c1.metric("Overall accuracy",f"{overall:.1f}%")
    c2.metric("Hard 8 accuracy",f"{hard_acc:.1f}%")
    c3.metric("Gap",f"{overall-hard_acc:.1f} pp")
    st.dataframe(pd.DataFrame({"query_id":qdf.query_id,"query":qdf.query,"correct":hits}), use_container_width=True)
