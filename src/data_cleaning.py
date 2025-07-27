import pandas as pd
import numpy as np
import string
import os
print("Current working directory:", os.getcwd())

# Load the dataset
df = pd.read_csv('../data/raw/AI_Human.csv')

# Simple print-out to see how big the dataset is and how many human/AI examples we have
#print('Total Texts:', df['generated'].count())  # Print total # of rows
#print('Human Written Texts:', (df['generated'] == 0.0).sum())  # Print # of human-written texts (label 0)
#print('AI Generated Texts:', (df['generated'] == 1.0).sum())  # Print # of AI-generated texts (label 1)

#---------------------------------Begin Cleaning Text---------------------------

# Remove missing text (NAs or rows where there is only an empty string)
def remove_missing(df):
    df = df.dropna(subset=['text'])  # Drop rows where the 'text' column is NaN
    df = df[df['text'].str.strip() != '']  # Drop rows where the 'text' column is empty
    return df

# Remove tags (e.g. newline and apostophres)
def remove_tags(text):
    tags = ['\n', '\'']
    for tag in tags:
        # Replace with an empty string instead of the tag
        text = text.replace(tag, '')
    return text

# Remove punctuation
def remove_punc(text):
    # Filter out all punctuation chars
    new_text = ''.join([char for char in text if char not in string.punctuation])
    return new_text

# Lowercase
def lowercase(text):
    new_text = text.lower()
    return new_text

# Main function that calls all our cleaning/filter functions above
def clean_text(df):
    print("Inside the clean text function")
    #print(df.head())
    df = remove_missing(df)  
    # Apply text cleaning functions
    df['text'] = df['text'].apply(remove_tags)
    df['text'] = df['text'].apply(remove_punc)
    df['text'] = df['text'].apply(lowercase)
    df = df.dropna(subset=['text', 'generated'])
    return df

# Apply the full cleaning process
df = clean_text(df)
print("Outside the clean text function")
#print(df.head())

# Save cleaned dataset for future use
df.to_csv('../data/cleaned/AI_Human_cleaned.csv', index=False)

# Checks if there are any empty rows after the cleaning
print(f'Empty Rows after cleaning: {(df["text"].str.strip() == "").sum()}')
