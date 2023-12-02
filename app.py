import re
import numpy as np
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
from flask import Flask, render_template, request
from sentence_transformers import SentenceTransformer, util
import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')


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
    lemmatizer = WordNetLemmatizer()
    lemmatized_keywords = lemmatize_keywords(keywords)

    # Iterate over each sentence
    for sentence in sentence_list:
        sentence = sentence.strip()

        # Tokenize the sentence into words
        tokens = word_tokenize(sentence)

        # Lemmatize the tokens using the get_wordnet_pos function
        tagged_tokens = pos_tag(tokens)
        lemmatized_tokens = [lemmatizer.lemmatize(token, get_wordnet_pos(pos)) for token, pos in tagged_tokens]
        lemmatized_sentence = " ".join(lemmatized_tokens)

        # Check if any of the lemmatized keywords are present in the lemmatized sentence
        if any(keyword.lower() in lemmatized_sentence.lower() for keyword in lemmatized_keywords):
            sentences.append(sentence)

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

sbert_model = SentenceTransformer('bert-base-nli-mean-tokens')

def search_sentences_with_bert(text, keywords, sbert_model):
    sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', text)
    sentences_with_keywords = []

    # Encode the keywords using the SBERT model
    keyword_embeddings = sbert_model.encode(keywords, convert_to_tensor=True)

    for sentence in sentences:
        # Encode the sentence using the SBERT model
        sentence_embedding = sbert_model.encode([sentence], convert_to_tensor=True)

        # Calculate cosine similarity between the sentence and keywords
        cosine_scores = util.pytorch_cos_sim(sentence_embedding, keyword_embeddings)
        max_score = cosine_scores.max()

        # You can adjust this threshold to control the similarity threshold
        if max_score > 0.7:
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
    matching_sentences_with_bert = search_sentences_with_bert(input_text, keywords, sbert_model)

    # Search for matching sentences without lemmatizer
    matching_sentences_without_lemmatizer = search_sentences_without_lemmatizer(input_text, keywords)

    if request.form['choice'] == '1':
        matching_sentences = matching_sentences_without_lemmatizer
        choice = '1'
    elif request.form['choice'] == '2':
        matching_sentences = matching_sentences_with_lemmatizer
        choice = '2'
    else:
        matching_sentences = matching_sentences_with_bert
        choice = '3'

    return render_template('index.html', sentences=matching_sentences, choice=choice)

if __name__ == '__main__':
    app.run(debug=True)
