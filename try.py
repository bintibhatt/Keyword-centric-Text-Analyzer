import re
import numpy as np
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
from flask import Flask, render_template, request
from sentence_transformers import SentenceTransformer, util

app = Flask(__name__)

# Load a pre-trained BERT model for sentence embeddings
bert_model = SentenceTransformer('bert-base-nli-mean-tokens')

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

def search_sentences(text, keywords, lemmatize=False, bert_search=False):
    sentences = []

    lemmatizer = WordNetLemmatizer()
    if lemmatize:
        keywords = [lemmatizer.lemmatize(keyword, get_wordnet_pos(pos_tag([keyword])[0][1])) for keyword in keywords]

    if bert_search:
        sentence_embeddings = bert_model.encode(text, convert_to_tensor=True)
        keyword_embeddings = bert_model.encode(keywords, convert_to_tensor=True)

        # Ensure that the number of embeddings matches the number of sentences
        if len(sentence_embeddings) != len(text):
            return sentences  # Handle this error gracefully

        for i, similarity in enumerate(similarities):
            if i < len(text) and any(similarity > 0.8):
                sentences.append(text[i])
    else:
        sentence_list = re.split(r'[.!?]+', text)

        for sentence in sentence_list:
            sentence = sentence.strip()
            sentence_lower = sentence.lower()
            if any(keyword.lower() in sentence_lower for keyword in keywords):
                sentences.append(sentence)

    return sentences

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/search', methods=['POST'])
def search():
    input_text = request.form['text']
    input_keywords = request.form['keywords'].split(',')

    # Remove leading/trailing whitespaces from the keywords
    keywords = [keyword.strip() for keyword in input_keywords]

    choice = request.form['choice']

    if choice == '1':
        matching_sentences = search_sentences(input_text, keywords, lemmatize=False, bert_search=False)
    elif choice == '2':
        matching_sentences = search_sentences(input_text, keywords, lemmatize=True, bert_search=False)
    elif choice == '3':
        matching_sentences = search_sentences(input_text, keywords, lemmatize=False, bert_search=True)
    else:
        matching_sentences = []  # Handle other choices as needed

    return render_template('index.html', sentences=matching_sentences, choice=choice)

if __name__ == '__main__':
    app.run(debug=True)
