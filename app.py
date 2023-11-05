import re
import numpy as np
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
from flask import Flask, render_template, request
from sentence_transformers import SentenceTransformer

app = Flask(__name__)

def get_wordnet_pos(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN

def lemmatize_keywords(keywords):
    lemmatizer = WordNetLemmatizer()
    lemmatized_keywords = []
    for keyword in keywords:
        tokens = word_tokenize(keyword)
        tagged_tokens = pos_tag(tokens)
        lemmatized_tokens = [lemmatizer.lemmatize(token, get_wordnet_pos(pos_tag)) for token, pos_tag in tagged_tokens]
        lemmatized_keyword = " ".join(lemmatized_tokens)
        lemmatized_keywords.append(lemmatized_keyword)
    return lemmatized_keywords

def search_sentences_with_lemmatizer(text, keywords):
    sentences = []
    paragraph = text

    # Split the paragraph into sentences
    sentence_list = re.split(r'[.!?]+', paragraph)

    # Lemmatize the keywords
    lemmatized_keywords = lemmatize_keywords(keywords)

    # Iterate over each sentence
    for sentence in sentence_list:
        sentence = sentence.strip()
        for keyword in lemmatized_keywords:
            if keyword.lower() in sentence.lower():
                sentences.append(sentence)
                break

    return sentences

def search_sentences_without_lemmatizer(text, keywords):
    # Split the text into sentences
    sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', text)

    # Initialize a list to store the sentences containing the keywords
    sentences_with_keywords = []

    for sentence in sentences:
        # Tokenize the sentence into words
        tokens = re.findall(r'\b\w+\b', sentence.lower())

        # Check if any of the keywords are present in the sentence
        if any(keyword.lower() in tokens for keyword in keywords):
            sentences_with_keywords.append(sentence)

    return sentences_with_keywords

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/search', methods=['POST'])
def search():
    input_text = request.form['text']
    input_keywords = request.form['keywords'].split(',')

    # Remove leading/trailing whitespaces from the keywords
    keywords = [keyword.strip() for keyword in input_keywords]

    # Search for matching sentences with lemmatizer
    matching_sentences_with_lemmatizer = search_sentences_with_lemmatizer(input_text, keywords)

    # Search for matching sentences with BERT
    # matching_sentences_with_bert = search_sentences_with_bert(input_text, keywords)

    # Search for matching sentences without lemmatizer
    matching_sentences_without_lemmatizer = search_sentences_without_lemmatizer(input_text, keywords)

    if request.form['choice'] == '1':
        matching_sentences = matching_sentences_without_lemmatizer
        choice = '1'
    elif request.form['choice'] == '2':
        matching_sentences = matching_sentences_with_lemmatizer
        choice = '2'
    else:
        # matching_sentences = matching_sentences_with_bert
        choice = '3'

    return render_template('index.html', sentences=matching_sentences, choice=choice)

if __name__ == '__main__':
    app.run(debug=True)
