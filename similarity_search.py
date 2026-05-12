# INSTALLING AND IMPORTING LIBRARIES
# !pip install sentence-transformers==4.1.0 | tail -n 1

import math
import numpy as np
import scipy
import torch
from sentence_transformers import SentenceTransformer

# Obtaining Vector Embeddings
documents = ['Bugs introduced by the intern had to be squashed by the lead developer.',
             'Bugs found by the quality assurance engineer were difficult to debug.',
             'Bugs are common throughout the warm summer months, according to the entomologist.',
             'Bugs, in particular spiders, are extensively studied by arachnologists.']

model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
embeddings = model.encode(documents)

# Metric 1: L2 (Euclidean) Distance
def euclidean_distance_fn(vector1, vector2):
    squared_sum = sum((x - y) ** 2 for x, y in zip(vector1, vector2))
    return math.sqrt(squared_sum)

# Manual Way - Inefficient although correct
l2_dist_manual = np.zeros([4, 4])
for i in range(embeddings.shape[0]):
    for j in range(embeddings.shape[0]):
        l2_dist_manual[i, j] = euclidean_distance_fn(embeddings[i], embeddings[j])

# Slightly Faster Way (More Computationally Efficient)
l2_dist_manual_improved = np.zeros([4, 4])
for i in range(embeddings.shape[0]):
    for j in range(embeddings.shape[0]):
        if j > i:
            l2_dist_manual_improved[i, j] = euclidean_distance_fn(embeddings[i], embeddings[j])
        elif i > j:
            l2_dist_manual_improved[i, j] = l2_dist_manual[j, i]

print(l2_dist_manual)
print(l2_dist_manual_improved)

# Fastest Way - Using Scipy
l2_dist_scipy = scipy.spatial.distance.cdist(embeddings, embeddings, 'euclidean')
print(l2_dist_scipy)

# Metric 2: Dot Product Similarity And Distance
def dot_product_fn(vector1, vector2):
    return sum(x * y for x, y in zip(vector1, vector2))

# Manual Way - Inefficient although correct
dot_product_manual = np.empty([4, 4])
for i in range(embeddings.shape[0]):
    for j in range(embeddings.shape[0]):
        dot_product_manual[i, j] = dot_product_fn(embeddings[i], embeddings[j])

# Faster Way (Using Matrix Multiplication)
dot_product_operator = embeddings @ embeddings.T

print(dot_product_manual)
print(dot_product_operator)

# Other Possible Ways - if both of the matrices we want to multiply are two-dimensional
np.matmul(embeddings, embeddings.T)
np.dot(embeddings, embeddings.T)

# Calculating Dot Product Distance (Negative of the dot product)
dot_product_distance = -dot_product_manual
print(dot_product_distance)

# Metric 3: Cosine Similarity And Distance

# Normalizing Embedding Vectors (Manually)
l2_norms = np.sqrt(np.sum(embeddings ** 2, axis = 1))
l2_norms_reshaped = l2_norms.reshape(-1, 1)
normalized_embeddings_manual = embeddings / l2_norms_reshaped
print(normalized_embeddings_manual)

# Normalizing Embedding Vectors (Using PyTorch)
normalized_embeddings_torch = torch.nn.functional.normalize(torch.from_numpy(embeddings)).numpy()
print(normalized_embeddings_torch)

# Calculating Cosine Similarity (Manually)
cosine_similarity_manual = np.empty([4, 4])
for i in range(normalized_embeddings_manual.shape[0]):
    for j in range(normalized_embeddings_manual.shape[0]):
        cosine_similarity_manual[i, j] = dot_product_fn(normalized_embeddings_manual[i], normalized_embeddings_manual[j])

# Calculating Cosine Similarity (Using Matrix Multiplication)
cosine_similarity_operator = normalized_embeddings_manual @ normalized_embeddings_manual.T

print(cosine_similarity_manual)
print(cosine_similarity_operator)

# Calculating Cosine Distance (1 minus the cosine similarity)
cosine_distance = 1 - cosine_similarity_manual
print(cosine_distance)

# Application: Similarity Search Using A Query
# Which of the 4 documents is most similar to the query "Who is responsible for a coding project and fixing others' mistakes?" using cosine similarity? 
query_embedding = model.encode(["Who is responsible for a coding project and fixing others' mistakes?"])
normalized_query_embedding = torch.nn.functional.normalize(torch.from_numpy(query_embedding)).numpy()
cosine_similarity_q3 = normalized_embeddings_manual @ normalized_query_embedding.T
highest_cossim_position = cosine_similarity_q3.argmax()
print(documents[highest_cossim_position])
