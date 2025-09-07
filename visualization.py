import streamlit as st
import pdfplumber, docx
from collections import Counter
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from textblob import TextBlob
import seaborn as sns
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
def extract_text(uploaded_file):
    text = ""
    if uploaded_file.name.endswith(".pdf"):
        with pdfplumber.open(uploaded_file) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + " "
    elif uploaded_file.name.endswith(".docx"):
        doc = docx.Document(uploaded_file)
        for para in doc.paragraphs:
            text += para.text + " "
    return text
st.title("📚 Text Visualization ")
uploaded_file = st.file_uploader("Upload PDF or Word file", type=["pdf", "docx"])
if uploaded_file:
    text = extract_text(uploaded_file)
    words = re.findall(r'\w+', text.lower())
    filtered_words = [w for w in words if w not in stop_words and len(w) > 1]
    word_counts = Counter(filtered_words)
    common_words = word_counts.most_common(10)
    top_words, top_counts = zip(*common_words)
    if st.button("📊 Show Bar Chart"):
        st.subheader("Top 10 Words - Bar Chart")
        fig, ax = plt.subplots()
        ax.bar(top_words, top_counts, color='green')
        ax.set_ylabel("Frequency")
        plt.xticks(rotation=45)
        st.pyplot(fig)
    if st.button("🥧 Show Pie Chart"):
        st.subheader("Top 10 Words - Pie Chart")
        fig, ax = plt.subplots()
        ax.pie(top_counts, labels=top_words, autopct='%1.1f%%', startangle=140)
        ax.axis("equal")
        st.pyplot(fig)
    if st.button("☁️ Show Word Cloud"):
        st.subheader("Word Cloud")
        wc = WordCloud(width=600, height=400, background_color="white").generate(" ".join(filtered_words))
        fig, ax = plt.subplots()
        ax.imshow(wc, interpolation="bilinear")
        ax.axis("off")
        st.pyplot(fig)
    if st.button("🔥 Show Heatmap"):
        st.subheader("Word Co-occurrence Heatmap")
        matrix = np.zeros((len(top_words), len(top_words)))
        for i, w1 in enumerate(top_words):
            for j, w2 in enumerate(top_words):
                matrix[i, j] = text.lower().count(f"{w1} {w2}")
        fig, ax = plt.subplots()
        sns.heatmap(matrix, xticklabels=top_words, yticklabels=top_words, cmap="YlGnBu", annot=True, fmt=".0f", ax=ax)
        st.pyplot(fig)
    if st.button("🧠 Show Sentiment Analysis"):
        st.subheader("Sentiment Analysis Score")
        blob = TextBlob(text)
        sentiment = blob.sentiment
        st.write(f"**Polarity:** {sentiment.polarity:.2f} (−1 = negative, +1 = positive)")
        st.write(f"**Subjectivity:** {sentiment.subjectivity:.2f} (0 = objective, 1 = subjective)")
        fig, ax = plt.subplots()
        ax.bar(["Polarity", "Subjectivity"], [sentiment.polarity, sentiment.subjectivity], color=["green", "orange"])
        ax.set_ylim(-1, 1)
        ax.set_ylabel("Score")
        ax.set_title("Sentiment Scores")
        st.pyplot(fig)
