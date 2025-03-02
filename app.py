import streamlit as st
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from string import punctuation
from collections import Counter
from heapq import nlargest

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

def summarize_text(text, num_sentences):
    """Summarizes the input text using word frequency analysis."""
    if not text.strip():
        return "Error: Please enter some text to summarize."
    
    # Tokenization and removing stopwords & punctuation
    doc = nlp(text)
    words = [token.text.lower() for token in doc if token.text.lower() not in STOP_WORDS and token.text not in punctuation]
    
    if not words:
        return "Error: No valid words found in the text."
    
    # Calculate word frequency
    word_freq = Counter(words)
    max_freq = max(word_freq.values())
    word_freq = {word: freq / max_freq for word, freq in word_freq.items()}  # Normalize frequency

    # Sentence tokenization and scoring
    sent_token = [sent.text for sent in doc.sents]
    sent_score = {}
    for sent in sent_token:
        for word in sent.lower().split():
            if word in word_freq:
                sent_score[sent] = sent_score.get(sent, 0) + word_freq[word]
    
    # Ensure num_sentences is within a valid range
    num_sentences = max(1, min(num_sentences, len(sent_token)))
    
    # Get the highest scoring sentences
    summarized_sentences = nlargest(num_sentences, sent_score, key=sent_score.get)
    
    return " ".join(summarized_sentences)

# Streamlit UI
st.title("Smart Text Summarizer")
st.write("A simple and basic text summarization tool.")

# Text input box
text = st.text_area("Enter the text you want to summarize:")

# Number of sentences selector
num_sentences = st.slider("Select number of sentences for summary:", min_value=1, max_value=10, value=3)

# Summarization button
if st.button("Summarize"):
    summary = summarize_text(text, num_sentences)
    st.subheader("Summary:")
    st.write(summary)
