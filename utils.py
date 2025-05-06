import os
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import re

# Setup stemmer
factory = StemmerFactory()
stemmer = factory.create_stemmer()

def load_documents(folder_path):
    documents = []
    filenames = []
    for filename in os.listdir(folder_path):
        if filename.endswith('.txt'):
            with open(os.path.join(folder_path, filename), 'r', encoding='utf-8') as f:
                documents.append(f.read())
                filenames.append(filename)
    return documents, filenames

stopwords_indonesia = set([
    "yang", "dan", "di", "ke", "dari", "ini", "untuk", "dengan", "pada", "adalah", 
    "itu", "dalam", "atau", "juga", "karena", "oleh", "sebagai", "saat", "tidak", "telah"
    # Tambahkan sesuai kebutuhan
])

def preprocess(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    words = text.split()
    words = [word for word in words if word not in stopwords_indonesia]
    text = ' '.join(words)
    text = stemmer.stem(text)
    return text

