"""
Cosine Similarity Retrieval System using Sentence Transformers

This module implements a document retrieval system using cosine similarity that:
1. Encodes documents using SentenceTransformers
2. Calculates cosine similarity between query and document embeddings
3. Ranks documents by similarity scores
"""

from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def main():
    """Main function to demonstrate the cosine similarity retrieval system."""
    
    # Sample documents dataset
    docs = [
        "Machine learning is amazing.",
        "Python is widely used for AI.",
        "Neural networks learn patterns.",
        "Data science includes ML and statistics.",
        "Deep learning uses neural networks.",
        "Natural language processing is fun.",
        "Supervised learning needs labeled data.",
        "Unsupervised learning finds hidden patterns.",
        "AI can automate tasks.",
        "Reinforcement learning uses rewards."
    ]
    
    print("Initializing Sentence Transformer model...")
    # Initialize the sentence transformer model
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    print("Encoding documents...")
    # Encode all documents to get embeddings
    doc_embeddings = model.encode(docs)  # Shape: (10, embedding_dim)
    print(f"Document embeddings shape: {doc_embeddings.shape}")
    
    # Query document
    query_doc = "I want to learn about neural networks."
    print(f"\nQuery: '{query_doc}'")
    
    # Encode query
    query_embedding = model.encode([query_doc])  # Shape: (1, embedding_dim)
    
    # Calculate cosine similarities between query and all documents
    similarities = cosine_similarity(query_embedding, doc_embeddings)[0]
    
    # Find the most similar document
    best_match_idx = np.argmax(similarities)
    best_similarity = similarities[best_match_idx]
    
    print(f"\nMost similar document: Doc {best_match_idx + 1}")
    print(f"Document text: '{docs[best_match_idx]}'")
    print(f"Cosine similarity score: {best_similarity:.4f}")
    
    # Sort documents by similarity (highest to lowest)
    sorted_indices = np.argsort(similarities)[::-1]
    
    print("\nAll Documents Ranked by Similarity:")
    print("-" * 70)
    print(f"{'Rank':<6} {'Doc #':<6} {'Similarity':<12} {'Document Text'}")
    print("-" * 70)
    
    for rank, idx in enumerate(sorted_indices, 1):
        doc_num = idx + 1
        similarity = similarities[idx]
        doc_text = docs[idx]
        print(f"{rank:<6} {doc_num:<6} {similarity:<12.4f} {doc_text}")
    
    # Calculate and display similarity statistics
    print(f"\nSimilarity Statistics:")
    print(f"Average similarity: {np.mean(similarities):.4f}")
    print(f"Standard deviation: {np.std(similarities):.4f}")
    print(f"Min similarity: {np.min(similarities):.4f}")
    print(f"Max similarity: {np.max(similarities):.4f}")


def find_similar_documents(query, docs, model=None, top_k=5):
    """
    Find the top-k most similar documents to a query.
    
    Args:
        query (str): The query text
        docs (list): List of document strings
        model: SentenceTransformer model (optional, will create if None)
        top_k (int): Number of top results to return
    
    Returns:
        list: List of tuples (doc_index, similarity_score, document_text)
    """
    if model is None:
        model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Encode documents and query
    doc_embeddings = model.encode(docs)
    query_embedding = model.encode([query])
    
    # Calculate similarities
    similarities = cosine_similarity(query_embedding, doc_embeddings)[0]
    
    # Get top-k results
    top_indices = np.argsort(similarities)[::-1][:top_k]
    
    results = []
    for idx in top_indices:
        results.append((idx, similarities[idx], docs[idx]))
    
    return results


if __name__ == "__main__":
    main()
    
    # Demonstrate the reusable function
    print("\n" + "="*70)
    print("USING THE REUSABLE FUNCTION:")
    print("="*70)
    
    docs = [
        "Machine learning is amazing.",
        "Python is widely used for AI.",
        "Neural networks learn patterns.",
        "Data science includes ML and statistics.",
        "Deep learning uses neural networks.",
        "Natural language processing is fun.",
        "Supervised learning needs labeled data.",
        "Unsupervised learning finds hidden patterns.",
        "AI can automate tasks.",
        "Reinforcement learning uses rewards."
    ]
    
    query = "I want to learn about neural networks."
    results = find_similar_documents(query, docs, top_k=3)
    
    print(f"Query: '{query}'")
    print(f"\nTop 3 most similar documents:")
    for i, (doc_idx, similarity, doc_text) in enumerate(results, 1):
        print(f"{i}. Doc {doc_idx + 1}: {similarity:.4f} - '{doc_text}'")