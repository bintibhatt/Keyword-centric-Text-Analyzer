
# Keyword Centric Text Analyzer

## Overview

This Flask web app searches for sentences with specific keywords. It offers three methods:

* Basic Search: Matches sentences without lemmatization.
* Lemmatized Search: Matches sentences after lemmatizing keywords.
* BERT Search: Matches sentences using BERT embeddings.



## SetUp

1. Install libraries: `pip install requirements.txt`
2. By running run_app.py, it will download:
* nltk.download('punkt')
* nltk.download('averaged_perceptron_tagger')
* nltk.download('wordnet')
3. Run: ` python your_app_name.py`
    
## Run Locally

Clone the project

```bash
  git clone https://github.com/bintibhatt/Keyword-centric-text-analyzer
```

Go to the project directory

```bash
  cd Keyword-centric-text-analyzer
```

Install dependencies

```bash
  pip install -r requirements.txt
```

Start the server

```bash
  python run_app.py
```

