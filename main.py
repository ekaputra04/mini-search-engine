from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from utils import load_documents, preprocess, highlight_keywords, stopwords_indonesia

# 1. Load dokumen
documents, filenames = load_documents("documents")

# 2. Preprocessing (dengan dan tanpa stopwords/stemming)
preprocessed_docs = []
for doc in documents:
    clean_text = preprocess(doc)  # pakai stopwords + stemming
    preprocessed_docs.append(clean_text)

# 3. Buat TF-IDF vectorizer
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(preprocessed_docs)

# 4. Input query dan jumlah hasil
query = input("Masukkan kata kunci pencarian: ")
limit = int(input("Ingin menampilkan berapa hasil teratas? "))
preprocessed_query = preprocess(query)
query_vec = vectorizer.transform([preprocessed_query])

# 5. Hitung cosine similarity
cos_sim = cosine_similarity(query_vec, tfidf_matrix)
ranking = cos_sim[0].argsort()[::-1]

# 6. Tampilkan hasil
print(f"\nHasil pencarian untuk: \"{query}\" (Top {limit})\n")
keywords = query.lower().split()

for i in ranking[:limit]:
    skor = cos_sim[0][i]
    if skor == 0:
        continue  # Lewati jika tidak relevan

    original = documents[i]
    highlighted = highlight_keywords(original, keywords)
    print(f"📄 {filenames[i]} (Skor: {skor:.4f}):\n{highlighted}\n")
