import pandas as pd
import numpy as np
import streamlit as st
from dotenv import load_dotenv

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings


# -----------------------------
# 1. Load environment variables
# -----------------------------
load_dotenv()


# -----------------------------
# 2. Load books dataset
# -----------------------------
@st.cache_data
def load_books():
    books_df = pd.read_csv("books_with_emotions.csv")
    books_df["large_thumbnail"] = books_df["thumbnail"] + "&fife=w800"
    books_df["large_thumbnail"] = np.where(
        books_df["large_thumbnail"].isna(),
        "cover-not-found.jpg",
        books_df["large_thumbnail"],
    )
    return books_df


books = load_books()


# -----------------------------
# 3. Load and split text data
# -----------------------------
@st.cache_resource
def load_vector_db():
    raw_documents = TextLoader("tagged_description.txt", encoding="utf-8").load()
    text_splitter = CharacterTextSplitter(separator="\n", chunk_size=500, chunk_overlap=0)
    documents = text_splitter.split_documents(raw_documents)

    # Create HuggingFace embeddings
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Create Chroma vector store
    db_books = Chroma.from_documents(documents, embedding=embeddings)
    return db_books


db_books = load_vector_db()


# -----------------------------
# 4. Define semantic retriever
# -----------------------------
def retrieve_semantic_recommendations(query, category=None, tone=None,
                                      initial_top_k=50, final_top_k=16):
    recs = db_books.similarity_search(query, k=initial_top_k)
    books_list = [int(rec.page_content.strip('"').split()[0]) for rec in recs]
    book_recs = books[books["isbn13"].isin(books_list)].head(initial_top_k)

    if category != "All":
        book_recs = book_recs[book_recs["simple_categories"] == category].head(final_top_k)
    else:
        book_recs = book_recs.head(final_top_k)

    # Emotion-based sorting
    tone_map = {
        "Happy": "joy",
        "Surprising": "surprise",
        "Angry": "anger",
        "Suspenseful": "fear",
        "Sad": "sadness"
    }
    if tone in tone_map:
        book_recs.sort_values(by=tone_map[tone], ascending=False, inplace=True)

    return book_recs


# -----------------------------
# 5. Streamlit UI
# -----------------------------
st.set_page_config(page_title="Book Recommender", layout="wide")

st.title("📚✨Book Recommender")
st.markdown("""
Find your next favorite read with **AI-powered recommendations**.  
Search by *story idea*, *category*, or even the *emotion* you want to feel!
""")

query = st.text_area("🔍 Describe a book you'd like to read:",
                     placeholder="e.g., A mystery novel set in Victorian London")

col1, col2 = st.columns(2)

categories = ["All"] + sorted(books["simple_categories"].unique())
tones = ["All", "Happy", "Surprising", "Angry", "Suspenseful", "Sad"]

with col1:
    category = st.selectbox("📂 Choose a category:", categories)

with col2:
    tone = st.selectbox("🎭 Choose an emotional tone:", tones)

submit = st.button("🚀 Get Recommendations")

# -----------------------------
# 6. Display recommendations
# -----------------------------
if submit and query:
    with st.spinner("Finding great reads for you..."):
        recommendations = retrieve_semantic_recommendations(query, category, tone)

    st.subheader("📖 Recommended Books")

    for _, row in recommendations.iterrows():
        with st.container():
            col_img, col_text = st.columns([1, 3])

            with col_img:
                st.image(row["large_thumbnail"], width=130)

            with col_text:
                authors_split = row["authors"].split(";")
                if len(authors_split) == 2:
                    authors_str = f"{authors_split[0]} and {authors_split[1]}"
                elif len(authors_split) > 2:
                    authors_str = f"{', '.join(authors_split[:-1])}, and {authors_split[-1]}"
                else:
                    authors_str = row["authors"]

                description = row["description"]
                truncated_desc_split = description.split()
                truncated_description = " ".join(truncated_desc_split[:30]) + "..."

                st.markdown(f"**{row['title']}** by *{authors_str}*")
                st.write(truncated_description)
                st.markdown("---")

elif submit:
    st.warning("Please enter a description or story idea first!")

