import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
txt_file_path = "AI_Back_Server/Project2/Sources/Sports_shoes_knowledge_base.txt"


with open(txt_file_path, "r", encoding="utf-8") as file:
    knowledge_base_text = file.read()
    
vectorizer = TfidfVectorizer()
vectorized_data = vectorizer.fit_transform(knowledge_base_text.splitlines())


# Convert to dataframe for saving
vectorized_df = pd.DataFrame(vectorized_data.toarray(), columns=vectorizer.get_feature_names_out())

# Save the vectorized data as JSON (simulating vector database storage)
vectorized_json_path = "AI_Back_Server/Project2/Sources/vectorized_data.json"
vectorized_df.to_json(vectorized_json_path, orient="records")

# Step 2: Convert to SQL format
entries = [line.split(": ", 1) for line in knowledge_base_text.splitlines() if ": " in line]
df = pd.DataFrame(entries, columns=["Category", "Content"])

# Generate SQL script for MySQL import
sql_statements = []
for _, row in df.iterrows():
    sql_statements.append(
        f"INSERT INTO knowledge_base (category, content) VALUES ('{row['Category']}', '{row['Content']}');"
    )

# Save SQL statements to a file
sql_script_path = "AI_Back_Server/Project2/Sources/knoledge_base.sql"
with open(sql_script_path, "w", encoding="utf-8") as file:
    file.write("\n".join(sql_statements))

# Output paths
print(f"Vectorized data saved to: {vectorized_json_path}")
print(f"SQL script saved to: {sql_script_path}")