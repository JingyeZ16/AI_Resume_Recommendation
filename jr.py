import pandas as pd
from transformers import AutoTokenizer, AutoModel
import torch
from sklearn.metrics.pairwise import cosine_similarity

# Load the datasets
job_dataset = pd.read_csv('jd.csv')
applicant_dataset = pd.read_csv('sample_candidates_100.csv')

# Display the first few rows of each dataset
# print(job_dataset.head())
# print(applicant_dataset.head())

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
model = AutoModel.from_pretrained("distilbert-base-uncased")


# Function to encode text
def encode_text(texts):
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        embeddings = model(**inputs).last_hidden_state.mean(dim=1)
    return embeddings


# Extract job descriptions and candidate profiles
job_descriptions = job_dataset['description'].tolist()
candidate_profiles = applicant_dataset['candidate_profile'].tolist()

# Encode job descriptions and candidate profiles
job_embeddings = encode_text(job_descriptions)
candidate_embeddings = encode_text(candidate_profiles)

# Compute cosine similarity
similarity_matrix = cosine_similarity(candidate_embeddings, job_embeddings)

# Create a DataFrame to display the results
similarity_df = pd.DataFrame(similarity_matrix, index=applicant_dataset['candidate_id'], columns=job_dataset['id'])


# Function to recommend jobs
def recommend_jobs(candidate_id, top_n=5):
    if candidate_id in similarity_df.index:
        similar_jobs = similarity_df.loc[candidate_id].nlargest(top_n)
        recommended_jobs = job_dataset[job_dataset['id'].isin(similar_jobs.index)]
        return recommended_jobs
    else:
        print(f"Candidate ID {candidate_id} not found in the dataset.")
        return pd.DataFrame()

# Recommend top 5 jobs for each candidate
for candidate_id in applicant_dataset['candidate_id']:
    print(f"Top 5 job recommendations for candidate ID {candidate_id}:")
    recommended_jobs = recommend_jobs(candidate_id, top_n=5)
    print(recommended_jobs)
