"""
Comparison between RandomForest and Cosine Similarity Retrieval Methods

This script compares the two different approaches for document retrieval:
1. RandomForest classifier approach (from main.py)
2. Direct cosine similarity approach (from cosine_similarity_retrieval.py)
"""

from sklearn.ensemble import RandomForestClassifier
from sentence_transformers import SentenceTransformer
import numpy as np
from scipy.stats import rankdata
from sklearn.metrics.pairwise import cosine_similarity


def main():
    """Compare both retrieval methods on the same dataset."""
    
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
    
    # Query document
    query_doc = "I want to learn about neural networks."
    
    print("=" * 80)
    print("DOCUMENT RETRIEVAL METHODS COMPARISON")
    print("=" * 80)
    print(f"Query: '{query_doc}'")
    print(f"Total documents: {len(docs)}")
    print()
    
    # Initialize model (shared between both methods)
    print("Loading SentenceTransformer model...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Encode documents and query
    doc_embeddings = model.encode(docs)
    query_embedding = model.encode([query_doc])
    
    print(f"Embedding dimension: {doc_embeddings.shape[1]}")
    print()
    
    # METHOD 1: RandomForest Classifier Approach
    print("=" * 50)
    print("METHOD 1: RandomForest Classifier")
    print("=" * 50)
    
    # Train RandomForest classifier
    labels = list(range(len(docs)))  # Classes 0-(n-1)
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(doc_embeddings, labels)
    
    # Predict
    rf_pred_class = clf.predict(query_embedding)[0]
    rf_pred_probs = clf.predict_proba(query_embedding)[0]
    
    # Convert to percentiles
    ranks = rankdata(rf_pred_probs, method='average')
    percentiles = 100 * (ranks - 1) / (len(rf_pred_probs) - 1)
    
    print(f"Top prediction: Doc {rf_pred_class + 1}")
    print(f"Document: '{docs[rf_pred_class]}'")
    print(f"Probability: {rf_pred_probs[rf_pred_class]:.4f}")
    print()
    
    # METHOD 2: Cosine Similarity Approach
    print("=" * 50)
    print("METHOD 2: Cosine Similarity")
    print("=" * 50)
    
    # Calculate cosine similarities
    similarities = cosine_similarity(query_embedding, doc_embeddings)[0]
    cs_best_idx = np.argmax(similarities)
    cs_best_similarity = similarities[cs_best_idx]
    
    print(f"Top prediction: Doc {cs_best_idx + 1}")
    print(f"Document: '{docs[cs_best_idx]}'")
    print(f"Similarity: {cs_best_similarity:.4f}")
    print()
    
    # DETAILED COMPARISON
    print("=" * 80)
    print("DETAILED COMPARISON - TOP 5 RESULTS")
    print("=" * 80)
    
    # Sort by RandomForest probabilities
    rf_sorted_indices = np.argsort(rf_pred_probs)[::-1]
    
    # Sort by cosine similarities
    cs_sorted_indices = np.argsort(similarities)[::-1]
    
    print(f"{'Rank':<4} {'RandomForest':<35} {'Cosine Similarity':<35}")
    print(f"{'':4} {'Doc | Prob':<35} {'Doc | Similarity':<35}")
    print("-" * 80)
    
    for i in range(5):
        # RandomForest results
        rf_idx = rf_sorted_indices[i]
        rf_prob = rf_pred_probs[rf_idx]
        rf_pct = percentiles[rf_idx]
        
        # Cosine similarity results
        cs_idx = cs_sorted_indices[i]
        cs_sim = similarities[cs_idx]
        
        print(f"{i+1:<4} {rf_idx+1:3} | {rf_prob:.3f} ({rf_pct:5.1f}%){'':<12} {cs_idx+1:3} | {cs_sim:.4f}")
    
    print()
    
    # ANALYSIS
    print("=" * 80)
    print("ANALYSIS")
    print("=" * 80)
    
    same_top_result = (rf_pred_class == cs_best_idx)
    print(f"Both methods agree on top result: {'YES' if same_top_result else 'NO'}")
    
    if same_top_result:
        print(f"Both methods identified Doc {rf_pred_class + 1} as most relevant:")
        print(f"  '{docs[rf_pred_class]}'")
    else:
        print(f"Different top results:")
        print(f"  RandomForest: Doc {rf_pred_class + 1} - '{docs[rf_pred_class]}'")
        print(f"  Cosine Sim:   Doc {cs_best_idx + 1} - '{docs[cs_best_idx]}'")
    
    print()
    print("Method Characteristics:")
    print("┌─ RandomForest Approach:")
    print("│  • Uses machine learning classifier")
    print("│  • Learns patterns in embedding space")
    print("│  • Provides probabilistic outputs")
    print("│  • Can capture complex non-linear relationships")
    print("│")
    print("└─ Cosine Similarity Approach:")
    print("   • Direct geometric similarity measure")
    print("   • Measures angle between embedding vectors")
    print("   • Intuitive and interpretable")
    print("   • Computationally simpler and faster")


if __name__ == "__main__":
    main()