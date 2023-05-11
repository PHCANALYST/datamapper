import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import spacy
import torch

# Load the GloVe word embedding model
nlp = spacy.load('en_core_web_md')

# Load the source file
source_df = pd.read_csv('target.csv')

# Load the upload file
upload_df = pd.read_csv('Source.csv')

# Calculate the vector representations of the descriptions in the source file
source_vectors = []
for description in source_df['Description']:
    # Create a spacy Doc object from the description
    doc = nlp(description)

    # Get the vector representation of the Doc object
    vector = doc.vector

    # Add the vector to the list of source vectors
    source_vectors.append(vector)

# Convert the list of source vectors to a NumPy array
source_vectors = np.array(source_vectors).reshape(len(source_vectors), -1)

# Iterate over the rows in the upload file and find the most similar value in the source file for each row
for index, row in upload_df.iterrows():
    upload_text = row['Text'].lower()

    # Create a spacy Doc object from the upload text
    doc = nlp(upload_text)

    # Get the vector representation of the Doc object
    upload_vector = doc.vector.reshape(1, -1)

    # Calculate the cosine similarity between the upload vector and the source vectors
    similarities = cosine_similarity(upload_vector, source_vectors)[0]

    # Find the index of the most similar source vector
    max_similarity_index = np.argmax(similarities)

    # Get the value of the most similar source vector
    max_similarity = similarities[max_similarity_index]

    # If the cosine similarity is greater than or equal to 0.5, print the upload text, the most similar source value, and the cosine similarity
    if max_similarity >= 0.5:
        mapped_value = source_df.iloc[max_similarity_index]['Attribute']
        print(f"Upload Value: row {index}: {upload_text} = Source value: {mapped_value}, Similarity: {max_similarity:.2%}")
        #(f" row {index} is likely plagiarized.")
    else:
        print(f"No similar value found for upload value in row {index}: {upload_text}")
        print(f"Upload value  {upload_text} :row {index}  {mapped_value} is likely original.")

# Create a PyTorch model
model = torch.nn.Sequential(
    torch.nn.Linear(len(source_vectors[0]), 100),
    torch.nn.ReLU(),
    torch.nn.Linear(100, 1),
    torch.nn.Sigmoid()
)

# Train the PyTorch model on a dataset of text
model.train(source_text)

# Use the PyTorch model to generate new text
new_text = model.generate(max_length=100)

# Print the generated text
print(new_text)
