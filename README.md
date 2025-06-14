# NER-Resume-Analyzer

## Description:
This project aims at training a Named Entity Recognition model with the help of spaCy (python package for Entity Recognition) and publicly available resume/job descriptions.
The trained model can then be loaded into python and be used for extracting 'user skills' from the input text.

## Training steps
The 'Training_Creation' directory contains 2 files. Running these in order will train the model
- [BIOCreation.py](/Training_Creation/BIOCreation.py) - Run this script to create the prepare the training data for the model. The file takes as input [Skills.json](/Training_Data/Skills.json) as well as [Resume.csv](/Training_Data/Resume.zip). Make sure to keep the 3 files in the same directory as to not face any error. The script will create a file title 'BIO_Resume.txt' to be further used for training the model.
- [OptimizedTrainingCreation](/Training_Creation/OptimizedTrainingCreation.py) - Run this script to train the Named Entity Recognition model. The script will take as input the previously created 'BIO_Resume.txt', the script then creates a 'train.spacy' file which is the format spaCy uses for training Entity Recogntion models. After the model is trained, it is stored in a directory named '/output/'. The model is successfully trained and can be used for skill extraction from text.

## Sample Usage
``
import spacy

def extract_skills(text):
  nlp=spacy.load('output')
    doc = nlp(text)
    skills=[]
    docs=nlp(text)
    for doc in docs.ents:
        skills.append(str.lower(doc.text))
    skills=list(set(skills))
    return st
``

## About Me:
Hi I'm Vansh Mahajan, An aspiring Data Scientist. You can connect with me on my Socials below:
<br><br>
[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://linkedin.com/in/vanshmahajan55/)

