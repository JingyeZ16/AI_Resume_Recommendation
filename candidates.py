import pandas as pd
candidates_data = {
    "candidate_id": list(range(1, 101)),
    "candidate_name": ["John Doe", "Jane Smith", "Emily Johnson", "Michael Brown", "Sarah Davis"] * 20,
    "candidate_profile": [
        "Skills: JavaScript, React, Node.js. Experience: 5 years as a web developer at XYZ Corp. Education: Bachelor's in Computer Science from University A.",
        "Skills: HTML, CSS, Angular, Python. Experience: 4 years as a software developer at ABC Inc. Education: Bachelor's in Software Engineering from University B.",
        "Skills: Java, Spring Boot, SQL, JavaScript. Experience: 6 years as a backend developer at DEF Ltd. Education: Master's in Computer Science from University C.",
        "Skills: PHP, Laravel, MySQL, HTML. Experience: 3 years as a full-stack developer at GHI LLC. Education: Bachelor's in Information Technology from University D.",
        "Skills: Python, Django, Machine Learning, Data Analysis. Experience: 4 years as a data scientist at JKL Co. Education: Bachelor's in Data Science from University E."
    ] * 20,
    "location": ["Houston,TX,US", "Austin,TX,US", "Dallas,TX,US", "San Antonio,TX,US", "Fort Worth,TX,US"] * 20,
    "job_preferences": ["Web Developer", "Software Developer", "Backend Developer", "Full-Stack Developer", "Data Scientist"] * 20
}

candidates_df = pd.DataFrame(candidates_data)

# Save the dataset to a CSV file
candidates_file_path = 'sample_candidates_100.csv'
candidates_df.to_csv(candidates_file_path, index=False)

candidates_file_path
