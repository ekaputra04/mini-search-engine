from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from utils import load_documents, preprocess

# 1. Load dokumen
documents, filenames = load_documents("documents")

# 2. Preprocessing
preprocessed_docs = [preprocess(doc) for doc in documents]

# 3. Buat vektor TF-IDF
vectorizer = TfidfVectorizer()  
tfidf_matrix = vectorizer.fit_transform(preprocessed_docs)

# 4. Input query user
query = input("Masukkan kata kunci pencarian: ")
preprocessed_query = preprocess(query)
query_vec = vectorizer.transform([preprocessed_query])

# 5. Hitung cosine similarity
cos_sim = cosine_similarity(query_vec, tfidf_matrix)

# 6. Tampilkan hasil pencarian
ranking = cos_sim[0].argsort()[::-1]

print(f"\nHasil pencarian untuk: \"{query}\"\n")
for i in ranking:
    print(f"{filenames[i]}: {documents[i]} (Skor: {cos_sim[0][i]:.4f})")
