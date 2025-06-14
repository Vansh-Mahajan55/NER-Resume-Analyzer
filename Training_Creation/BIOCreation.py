import pandas as pd
import json
import spacy

# Load the Resume.csv file
df = pd.read_csv("Resume.csv")

# Load the Skills.json file
with open("Skills.json", "r") as file:
    skills_data = json.load(file)

# Extract skill list and normalize (lowercase for case-insensitive matching)
skill_list = set(map(str.lower, skills_data["skill_list"]))

# Load SpaCy English model (Make sure it's installed: `python -m spacy download en_core_web_sm`)
nlp = spacy.load("en_core_web_sm")

# Function to convert text to BIO format
def text_to_bio(text, skills):
    tokens = [token.text for token in nlp(text)]  # Tokenize text
    labels = ["O"] * len(tokens)  # Default label is "O" (outside entity)

    i = 0
    while i < len(tokens):
        token_lower = tokens[i].lower()
        found = None

        # Check for multi-word skills (longest match first)
        for length in range(4, 0, -1):  # Check up to 4-word skills
            phrase = " ".join(tokens[i:i+length]).lower()
            if phrase in skills:
                found = (i, i + length, phrase)
                break
        
        if found:
            start, end, phrase = found
            labels[start] = "B-SKILL"  # Beginning of skill
            for j in range(start + 1, end):
                labels[j] = "I-SKILL"  # Inside skill
            i = end  # Move index forward
        else:
            i += 1

    return list(zip(tokens, labels))

# Process all resumes
bio_output = []
for resume in df["Resume_str"]:
    bio_data = text_to_bio(resume, skill_list)
    bio_output.append("\n".join([f"{token} {label}" for token, label in bio_data]) + "\n\n")

# Save to file
with open("BIO_Resume.txt", "w", encoding="utf-8") as f:
    f.writelines(bio_output)

print("BIO-formatted data saved as 'BIO_Resume.txt'.")
