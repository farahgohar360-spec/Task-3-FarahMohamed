import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


catalog_data = {
    'item_id': [101, 102, 103, 104, 105, 106, 107, 108],
    'title': [
        'Advanced Java Programming Masterclass',
        'Mobile App Architecture with Flutter',
        'Data Science Fundamentals and Neural Networks',
        'UI/UX Design Patterns & Figma Prototyping',
        'Building E-Commerce Marketplaces with Flutter',
        'Spring Boot Microservices and Java API Design',
        'Automotive Database Systems & Architecture',
        'Introduction to Algorithm Optimization in Java'
    ],
    'tags': [
        'java backend software development programming algorithms OOP code',
        'flutter mobile cross-platform widgets mobile application frontend dart',
        'data-science neural-networks machine-learning artificial-intelligence tensors python optimization',
        'ui ux design figma frontend user-interface prototyping typography graphics',
        'flutter mobile application e-commerce database dart backend cross-platform',
        'java spring-boot backend microservices api architecture software development Rest-API',
        'automotive hardware architecture database systems sql structured-data e-commerce',
        'java algorithms optimization performance patterns backend engineering data-structures'
    ]
}

df_items = pd.DataFrame(catalog_data)
print("--- Step 1: Mock Catalog Database Successfully Created ---")
print(f"Total available items to recommend: {len(df_items)}\n")

print("--- Step 2: Formulating User Interaction Profile ---")
user_interests = "I am focusing on software development mobile applications and backend engineering using java and flutter frameworks"
print(f"Captured User Profile Input:\n   \"{user_interests}\"\n")

print("--- Step 3: Mapping Text Tags to TF-IDF Numerical Vectors ---")
tfidf_vectorizer = TfidfVectorizer(stop_words='english')
all_documents = list(df_items['tags']) + [user_interests]
tfidf_matrix = tfidf_vectorizer.fit_transform(all_documents)
items_vector_matrix = tfidf_matrix[:-1]  
user_profile_vector = tfidf_matrix[-1:]
print(f"Vocabulary matrix shape: {items_vector_matrix.shape} (Matches items and unique tag terms)\n")

print("--- Step 4: Computing Geometric Alignment (Similarity Matrix) ---")
similarity_scores = cosine_similarity(user_profile_vector, items_vector_matrix).flatten()
df_items['similarity_score'] = similarity_scores

print("--- Step 5: Generating Truncated Top-N Recommendations ---")
top_n = 3
recommended_items = df_items.sort_values(by='similarity_score', ascending=False).head(top_n)

print(f"\n===== TOP {top_n} PERSONALIZED RECOMMENDATIONS FOR YOU =====")
for index, row in recommended_items.iterrows():
    print(f"\n✨ Item ID: {row['item_id']}")
    print(f"   Title: {row['title']}")
    print(f"   Match Score: {row['similarity_score'] * 100:.2f}%")
    print(f"   Associated Tags: {row['tags'][:80]}...")
print("==========================================================")