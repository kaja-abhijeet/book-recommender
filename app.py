import pandas as pd
import numpy as np
import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# Load environment variables (if any)
load_dotenv()

st.set_page_config(page_title="📚 Semantic Book Recommender", layout="wide")

# ---------------------------------
# Load Data
# ---------------------------------
@st.cache_data
def load_data():
    books = pd.read_csv("books_with_emotions.csv")
    books["large_thumbnail"] = books["thumbnail"] + "&fife=w800"
    books["large_thumbnail"] = np.where(
        books["large_thumbnail"].isna(),
        "cover-not-found.jpg",
        books["large_thumbnail"],
    )
    return books

books = load_data()

# ---------------------------------
# Load and prepare documents
# ---------------------------------
@st.cache_resource
def setup_embeddings():
    raw_documents = TextLoader("tagged_description.txt", encoding="utf-8").load()
    text_splitter = CharacterTextSplitter(separator="\n", chunk_size=500, chunk_overlap=0)
    documents = text_splitter.split_documents(raw_documents)

    huggingface_embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db_books = Chroma.from_documents(documents, embedding=huggingface_embeddings)
    return db_books

db_books = setup_embeddings()

# ---------------------------------
# Semantic Recommendation Logic
# ---------------------------------
def retrieve_semantic_recommendations(query, category=None, tone=None, initial_top_k=50, final_top_k=16):
    recs = db_books.similarity_search(query, k=initial_top_k)
    books_list = [int(rec.page_content.strip('"').split()[0]) for rec in recs]
    book_recs = books[books["isbn13"].isin(books_list)].head(initial_top_k)

    if category != "All":
        book_recs = book_recs[book_recs["simple_categories"] == category].head(final_top_k)
    else:
        book_recs = book_recs.head(final_top_k)

    if tone == "Happy":
        book_recs = book_recs.sort_values(by="joy", ascending=False)
    elif tone == "Surprising":
        book_recs = book_recs.sort_values(by="surprise", ascending=False)
    elif tone == "Angry":
        book_recs = book_recs.sort_values(by="anger", ascending=False)
    elif tone == "Suspenseful":
        book_recs = book_recs.sort_values(by="fear", ascending=False)
    elif tone == "Sad":
        book_recs = book_recs.sort_values(by="sadness", ascending=False)

    return book_recs

# ---------------------------------
# Streamlit UI
# ---------------------------------
st.title("📚✨ Book Recommender")
st.markdown("""
Find your next favorite read with **AI-powered recommendations**.  
Search by *story idea*, *category*, or even the *emotion* you want to feel!
""")

st.divider()

col1, col2, col3 = st.columns([3, 1.5, 1.5])

with col1:
    query = st.text_area("🔍 Describe a book you’d like to read:", placeholder="e.g., A mystery novel set in Victorian London")

with col2:
    categories = ["All"] + sorted(books["simple_categories"].unique())
    category = st.selectbox("📂 Choose a category:", categories)

with col3:
    tones = ["All", "Happy", "Surprising", "Angry", "Suspenseful", "Sad"]
    tone = st.selectbox("🎭 Choose an emotional tone:", tones)

st.divider()

if st.button("🚀 Get Recommendations"):
    if not query.strip():
        st.warning("Please enter a description or story idea first.")
    else:
        st.write("### 📖 Recommended Books")
        recs = retrieve_semantic_recommendations(query, category, tone)

        if recs.empty:
            st.info("No recommendations found. Try adjusting your query or filters.")
        else:
            cols = st.columns(4)
            for i, (_, row) in enumerate(recs.iterrows()):
                with cols[i % 4]:
                    st.image(row["large_thumbnail"], use_container_width=True)
                    st.markdown(f"**{row['title']}**")
                    st.caption(row["authors"])
                    desc = " ".join(row["description"].split()[:30]) + "..."
                    st.write(desc)
