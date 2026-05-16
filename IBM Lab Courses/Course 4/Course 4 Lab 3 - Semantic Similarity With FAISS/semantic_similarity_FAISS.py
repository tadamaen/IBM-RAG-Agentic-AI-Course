# INSTALLING AND IMPORTING PACKAGES
# !pip install faiss-cpu numpy scikit-learn
# !pip install "tensorflow>=2.0.0"
# !pip install --upgrade tensorflow-hub

import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import faiss
import re
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from pprint import pprint

# Suppressing warnings
def warn(*args, **kwargs):
    pass

import warnings
warnings.warn = warn
warnings.filterwarnings('ignore')

# Importing The 20 Newsgroups Dataset
from sklearn.datasets import fetch_20newsgroups
newsgroups_train = fetch_20newsgroups(subset = 'train')
pprint(list(newsgroups_train.target_names))

# Display the first 3 posts from the dataset
for i in range(3):
    print(f"Sample post {i+1}:\n")
    pprint(newsgroups_train.data[i])
    print("\n" + "-"*80 + "\n")

# Preprocessing The Data
newsgroups = fetch_20newsgroups(subset='all')
documents = newsgroups.data

def preprocess_text(text):
    text = re.sub(r'^From:.*\n?', '', text, flags = re.MULTILINE)
    text = re.sub(r'\S*@\S*\s?', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = text.lower()
    text = re.sub(r'\s+', ' ', text).strip()
    return text

processed_documents = [preprocess_text(doc) for doc in documents]

# Choose a sample post to display
sample_index = 0
print("Original post:\n")
print(newsgroups_train.data[sample_index])
print("\n" + "-"*80 + "\n")
print()
print("Preprocessed post:\n")
print(preprocess_text(newsgroups_train.data[sample_index]))
print("\n" + "-"*80 + "\n")

# Universal Sentence Encoder
embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
def embed_text(text):
    return embed(text).numpy()
X_use = np.vstack([embed_text([doc]) for doc in processed_documents])

# Indexing With FAISS
dimension = X_use.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(X_use)

# Function to perform a query using the Faiss index
def search(query_text, k = 5):
    preprocessed_query = preprocess_text(query_text)
    query_vector = embed_text([preprocessed_query])
    distances, indices = index.search(query_vector.astype('float32'), k)
    return distances, indices

query_text = "motorcycle"
distances, indices = search(query_text)
for i, idx in enumerate(indices[0]):
    print(f"Rank {i+1}: (Distance: {distances[0][i]})\n{processed_documents[idx]}\n")

for i, idx in enumerate(indices[0]):
    print(f"Rank {i+1}: (Distance: {distances[0][i]})\n{documents[idx]}\n")
