import re
import nltk
from nltk.corpus import stopwords
import spacy


try:
    nltk.data.find('corpora/stopwords')
except LookupError: 
    print("NLTK stopwords not found. Downloading...")
    nltk.download('stopwords')
except Exception as e:
    print(f"An unexpected error occurred during NLTK stopwords check/download: {e}")



german_stopwords = set(stopwords.words('german'))


try:
    nlp = spacy.load("de_core_news_sm")
except OSError:
    print("SpaCy German model 'de_core_news_sm' not found. Downloading...")
    spacy.cli.download("de_core_news_sm")
    nlp = spacy.load("de_core_news_sm")

def preprocess_text(text):
    
    if not isinstance(text, str):
        return "" 

    text = text.lower()
  
    text = re.sub(r'[^a-zäöüß\s]', '', text) 
    tokens = text.split() 

    
    lemmatized_tokens = []
    if tokens: 
        doc = nlp(" ".join(tokens)) 
        for token in doc:
            if token.text not in german_stopwords:
                lemmatized_tokens.append(token.lemma_)

    return " ".join(lemmatized_tokens)

