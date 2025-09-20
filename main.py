"""
ML Retrieval System using RandomForest and Sentence Transformers

This module implements a machine learning-based document retrieval system that:
1. Encodes documents using SentenceTransformers
2. Trains a RandomForest classifier to predict document relevance
3. Provides ranking and percentile scoring for query results
"""

from sklearn.ensemble import RandomForestClassifier
from sentence_transformers import SentenceTransformer
import numpy as np
from scipy.stats import rankdata


def main():
    """Main function to demonstrate the ML retrieval system."""
    
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
    
    # Create labels (classes 0-9)
    labels = list(range(10))
    
    print("Initializing Sentence Transformer model...")
    # Initialize the sentence transformer model
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    print("Encoding documents...")
    # Encode all documents to get embeddings
    doc_embeddings = model.encode(docs)  # Shape: (10, embedding_dim)
    print(f"Document embeddings shape: {doc_embeddings.shape}")
    
    print("Training RandomForest classifier...")
    # Train RandomForest classifier
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(doc_embeddings, labels)
    
    # Query document
    query_doc = "I want to learn about neural networks."
    print(f"\nQuery: '{query_doc}'")
    
    # Encode query
    query_embedding = model.encode([query_doc])  # Shape: (1, embedding_dim)
    
    # Predict closest document
    pred_class = clf.predict(query_embedding)[0]
    pred_probs = clf.predict_proba(query_embedding)[0]
    
    print(f"\nPredicted closest doc: Doc {pred_class + 1}")
    print(f"Predicted document text: '{docs[pred_class]}'")
    
    # Convert probabilities to percentiles/ranks
    ranks = rankdata(pred_probs, method='average')
    percentiles = 100 * (ranks - 1) / (len(pred_probs) - 1)
    
    print("\nDetailed Results:")
    print("-" * 60)
    print(f"{'Doc #':<6} {'Probability':<12} {'Percentile':<12} {'Document Text'}")
    print("-" * 60)
    
    # Sort by probability for better visualization
    sorted_indices = np.argsort(pred_probs)[::-1]
    
    for idx in sorted_indices:
        doc_num = idx + 1
        prob = pred_probs[idx]
        pct = percentiles[idx]
        doc_text = docs[idx]
        print(f"{doc_num:<6} {prob:<12.3f} {pct:<12.1f} {doc_text}")


if __name__ == "__main__":
    main()