
import csv
import nltk
from nltk.corpus import wordnet
from difflib import SequenceMatcher
import nltk

nltk.download('punkt')
nltk.download('wordnet')

# Open the source CSV file and read its data
source_data = []
with open('source_text.csv', 'r') as source_file:
    source_reader = csv.reader(source_file)
    for row in source_reader:
        source_data.append(row)

# Open the upload CSV file and read its data
upload_data = []
with open('uploaded_text.csv', 'r') as upload_file:
    upload_reader = csv.reader(upload_file)
    for row in upload_reader:
        upload_data.append(row)

# Initialize a list to store the similarities between the lines in the upload CSV and the source CSV
similarities = []

# Compare each line in the upload CSV with the lines in the source CSV
for i in range(len(upload_data)):
    upload_line = upload_data[i]
    for j in range(len(source_data)):
        source_line = source_data[j]

        # Tokenize the cells in the source line
        source_words = [word.lower() for cell in source_line for word in nltk.word_tokenize(cell)]

        # Calculate the similarity between the upload line and the source line
        similarity = 0
        for cell in upload_line:
            # Tokenize the words in the upload cell
            upload_words = nltk.word_tokenize(cell.lower())

            # Calculate the similarity between the upload cell and the source words
            cell_similarity = 0
            for source_word in source_words:
                for upload_word in upload_words:
                    if source_word == upload_word:
                        cell_similarity += 1
                    else:
                        source_synonyms = set([lemma.name() for syn in wordnet.synsets(source_word) for lemma in syn.lemmas()])
                        upload_synonyms = set([lemma.name() for syn in wordnet.synsets(upload_word) for lemma in syn.lemmas()])
                        if len(source_synonyms.intersection(upload_synonyms)) > 0:
                            cell_similarity += 1
                        elif SequenceMatcher(None, source_word, upload_word).ratio() > 0.8:
                            cell_similarity += 1
            
            # Calculate the similarity percentage for the upload cell
            cell_similarity_percentage = cell_similarity / max(len(source_words), len(upload_words))
            similarity += cell_similarity_percentage

        # Calculate the similarity percentage for the upload line
        similarity_percentage = similarity / len(upload_line) * 100

        # If the similarity percentage is more than 60%, add it to the list of similarities
        if similarity_percentage > 30:
            similarities.append((i+1, j+1, similarity_percentage))

# Output the similarities to the console
# Group similarities by upload line
grouped_similarities = {}
for similarity in similarities:
    upload_line_num, source_line_num, similarity_percentage = similarity
    if upload_line_num not in grouped_similarities:
        grouped_similarities[upload_line_num] = []
    grouped_similarities[upload_line_num].append((source_line_num, similarity_percentage))

# Output the similarities to the console
if grouped_similarities:
    for upload_line_num, similarities in grouped_similarities.items():
        upload_line = ' '.join(upload_data[upload_line_num-1])
        print(f"\nSimilarity for line {upload_line_num} : {upload_line}")
        for source_line_num, similarity_percentage in similarities:
            source_line = ' '.join(source_data[source_line_num-1])
            print(f"    - Line {source_line_num} : {source_line} | ({similarity_percentage:.2f}%)")
else:
    print("No similarities found.")

