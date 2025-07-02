import json
import pandas as pd
# Example JSON data (truncated for brevity)
json_data = '''
{
    "status": "OK",
    "request_id": "8739be65-eeab-43b6-859b-ccc6ec8b77e1",
    "parameters": {
        "query": "web developer in texas usa",
        "page": 1,
        "num_pages": 1
    },
    "data": [
        {
            "employer_name": "Dice",
            "job_title": "Web Developer - 6-month Contract - Houston Hybrid",
            "job_description": "An established energy client of mine is looking for an experienced Web Developer to join their team on an initial 6-month contract. ...",
            "job_city": "Houston",
            "job_state": "TX",
            "job_country": "US"
        },
        {
            "employer_name": "Charles Schwab",
            "job_title": "Software Web Developer",
            "job_description": "Your Opportunity ...",
            "job_city": "Austin",
            "job_state": "TX",
            "job_country": "US"
        }
        // More job data...
    ]
}
'''

# Load JSON data
data = json.loads(json_data)

# Extract job data
jobs = data['data']

# Convert to DataFrame for easier manipulation
job_df = pd.DataFrame(jobs)

from transformers import AutoTokenizer, AutoModel
import torch

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
model = AutoModel.from_pretrained("distilbert-base-uncased")


# To build a job recommendation system using Hugging Face, (job descriptions). 
# The steps involve loading the data, processing it to extract relevant information, encoding the text using Hugging Face's transformer
# models, and computing similarities to recommend jobs. Here's an example implementation:

# Step 1: Setup and Installation
# First, ensure you have the necessary libraries installed:

# pip install transformers torch pandas
# Step 2: Load and Parse JSON Data
# Load the JSON dataset and parse it to extract job descriptions and other relevant information.


import json
import pandas as pd
# Important : http request file json (get/ post depands on API, API format)
# Example JSON data (truncated for brevity)
json_data = '''
{
    "status": "OK",
    "request_id": "8739be65-eeab-43b6-859b-ccc6ec8b77e1",
    "parameters": {
        "query": "web developer in texas usa",
        "page": 1,
        "num_pages": 1
    },
    "data": [
        {
            "employer_name": "Dice",
            "job_title": "Web Developer - 6-month Contract - Houston Hybrid",
            "job_description": "An established energy client of mine is looking for an experienced Web Developer to join their team on an initial 6-month contract. ...",
            "job_city": "Houston",
            "job_state": "TX",
            "job_country": "US"
        },
        {
            "employer_name": "Charles Schwab",
            "job_title": "Software Web Developer",
            "job_description": "Your Opportunity ...",
            "job_city": "Austin",
            "job_state": "TX",
            "job_country": "US"
        }
        // More job data...
    ]
}
'''

# Load JSON data
data = json.loads(json_data)

# Extract job data
jobs = data['data']

# Convert to DataFrame for easier manipulation
job_df = pd.DataFrame(jobs)
# Step 3: Load a Pretrained Model. Load a pretrained model and tokenizer from Hugging Face.

# python
from transformers import AutoTokenizer, AutoModel
import torch

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
model = AutoModel.from_pretrained("distilbert-base-uncased")
# Step 4: Encode Job Descriptions
# Encode the job descriptions into embeddings using the transformer model.


def encode_text(texts):
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        embeddings = model(**inputs).last_hidden_state.mean(dim=1)
    return embeddings

# Encode job descriptions
job_embeddings = encode_text(job_df['job_description'].tolist())

from sklearn.metrics.pairwise import cosine_similarity

# Compute cosine similarity
similarity_matrix = cosine_similarity(job_embeddings)

# Convert similarity matrix to DataFrame for easier manipulation
similarity_df = pd.DataFrame(similarity_matrix, index=job_df['job_title'], columns=job_df['job_title'])

def recommend_jobs(job_title, top_n=5):
    similar_jobs = similarity_df[job_title].nlargest(top_n + 1).iloc[1:]  # Exclude the job itself
    return job_df[job_df['job_title'].isin(similar_jobs.index)]

# Example: Recommend top 5 jobs similar to "Web Developer - 6-month Contract - Houston Hybrid"
recommended_jobs = recommend_jobs("Web Developer - 6-month Contract - Houston Hybrid", top_n=5)
print(recommended_jobs)
