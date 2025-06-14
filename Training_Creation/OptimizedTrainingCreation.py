import spacy
from spacy.tokens import Doc, DocBin
from spacy.training import Example
import random

# Load BIO-formatted data from file
def load_bio_data(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read().strip().split("\n\n")  # Split resumes
    data = []
    
    for resume in content:
        tokens, labels = [], []
        for line in resume.split("\n"):
            parts = line.split()
            if len(parts) == 2:
                token, label = parts
                tokens.append(token)
                labels.append(label)
        
        data.append((tokens, labels))
    
    return data

# Convert BIO data to SpaCy format
def convert_to_spacy_format(bio_data, nlp):
    db = DocBin()
    for tokens, labels in bio_data:
        doc = Doc(nlp.vocab, words=tokens)
        entities = []
        start_char = 0
        
        for i, token in enumerate(tokens):
            end_char = start_char + len(token)
            
            # Check if token is part of an entity
            if labels[i] == "B-SKILL":
                entity_start = start_char
                entity_end = end_char
                
                # Check for multi-word skills (look ahead)
                for j in range(i + 1, len(tokens)):
                    if labels[j] == "I-SKILL":
                        entity_end = start_char + len(" ".join(tokens[i:j+1]))
                    else:
                        break
                
                span = doc.char_span(entity_start, entity_end, label="SKILL")
                if span:
                    entities.append(span)
            
            start_char = end_char + 1  # Move to next word position
        
        doc.set_ents(entities)
        db.add(doc)
    
    return db

# Initialize blank SpaCy model
nlp = spacy.blank("en")

# Load BIO data
bio_data = load_bio_data("BIO_Resume.txt")

# Convert to SpaCy format
doc_bin = convert_to_spacy_format(bio_data, nlp)

# Save training data
doc_bin.to_disk("train.spacy")

print("Training data saved as 'train.spacy'.")

# --- TRAINING THE MODEL ---

# Load blank English model
nlp = spacy.blank("en")

# Add Named Entity Recognition (NER) component
ner = nlp.add_pipe("ner", last=True)

# Add "SKILL" label
ner.add_label("SKILL")

# Load training data
db = DocBin().from_disk("train.spacy")
train_data = list(db.get_docs(nlp.vocab))

# Prepare examples for training
examples = []
for doc in train_data:
    entities = [(ent.start_char, ent.end_char, "SKILL") for ent in doc.ents]
    examples.append(Example.from_dict(doc, {"entities": entities}))

# Initialize optimizer
optimizer = nlp.create_optimizer()  # Corrected optimizer initialization
optimizer.learn_rate = 0.001  # Learning rate tuning
optimizer.L2 = 1e-6  # L2 Regularization (prevents overfitting)

# Training loop with optimizer
n_epochs = 20  # Increased epochs for better training
dropout_rate = 0.3  # Regularization to prevent overfitting
batch_size = 16  # Mini-batch size

for epoch in range(n_epochs):
    random.shuffle(examples)
    losses = {}

    for batch in spacy.util.minibatch(examples, size=batch_size):
        nlp.update(
            batch,
            drop=dropout_rate,  # Apply dropout
            losses=losses,
            sgd=optimizer  # Use improved Adam optimizer
        )
    
    print(f"Epoch {epoch+1}, Loss: {losses['ner']:.4f}")

# Save trained model
nlp.to_disk("output")

print("Training completed. Model saved in './output'.")
