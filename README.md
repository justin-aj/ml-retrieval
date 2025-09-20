# ML Retrieval System

An experimental document retrieval system exploring two different approaches to semantic similarity using modern NLP techniques. This project demonstrates how machine learning can be applied to information retrieval tasks through vector embeddings and similarity calculations.

## 🎯 Project Scope & Purpose

This project serves as a comparative study and proof-of-concept for document retrieval systems, specifically:

- **Research Focus**: Comparing traditional ML classification vs. direct similarity measurement for document retrieval
- **Educational Value**: Understanding how sentence embeddings can be leveraged for semantic search
- **Scalability Testing**: Evaluating different approaches for potential large-scale applications
- **Performance Analysis**: Benchmarking computational efficiency and accuracy trade-offs

## 🚀 Core Approaches

### 1. RandomForest Classification Approach
**Concept**: Treats document retrieval as a multi-class classification problem where each document is a separate class.

**How it works**:
- Train a RandomForest classifier on document embeddings with document IDs as labels
- For queries, predict which "document class" the query belongs to
- Provides probabilistic confidence scores for all documents

**Advantages**:
- Can learn complex non-linear patterns in embedding space
- Provides probabilistic outputs with confidence intervals
- May capture subtle semantic relationships through ensemble learning

**Limitations**:
- Requires training phase for each document set
- Not truly scalable for dynamic document collections
- Artificial classification problem (documents aren't natural classes)

### 2. Cosine Similarity Approach
**Concept**: Direct geometric similarity measurement between query and document vectors in embedding space.

**How it works**:
- Calculate cosine similarity between query embedding and all document embeddings
- Rank documents by similarity scores
- No training required - pure mathematical computation

**Advantages**:
- Mathematically intuitive and interpretable
- No training overhead - immediate deployment
- Naturally handles dynamic document collections
- Computationally efficient for real-time queries

**Limitations**:
- Limited to linear relationships in embedding space
- Cannot learn domain-specific similarity patterns

## 🔧 Technical Features

- Document encoding using SentenceTransformers (all-MiniLM-L6-v2 model)
- Parallel comparison of both retrieval methods
- Comprehensive performance metrics and analysis
- Reusable API for integration into larger systems

## Requirements

- Python 3.10+
- scikit-learn
- sentence-transformers
- numpy
- scipy

## 📊 Actual Output Results

Both methods were tested with the same query: **"I want to learn about neural networks."**

### Method Comparison Output
```
================================================================================
DOCUMENT RETRIEVAL METHODS COMPARISON
================================================================================
Query: 'I want to learn about neural networks.'
Total documents: 10

Loading SentenceTransformer model...
Embedding dimension: 384

==================================================
METHOD 1: RandomForest Classifier
==================================================
Top prediction: Doc 3
Document: 'Neural networks learn patterns.'
Probability: 0.1500

==================================================
METHOD 2: Cosine Similarity
==================================================
Top prediction: Doc 3
Document: 'Neural networks learn patterns.'
Similarity: 0.6648

================================================================================
DETAILED COMPARISON - TOP 5 RESULTS
================================================================================
Rank RandomForest                        Cosine Similarity
     Doc | Prob                          Doc | Similarity
--------------------------------------------------------------------------------
1      3 | 0.150 (100.0%)               3 | 0.6648
2      5 | 0.140 ( 88.9%)               5 | 0.5775
3      1 | 0.130 ( 77.8%)               1 | 0.4631
4      8 | 0.120 ( 66.7%)               2 | 0.4431
5     10 | 0.110 ( 55.6%)               9 | 0.4012

================================================================================
ANALYSIS
================================================================================
Both methods agree on top result: YES
Both methods identified Doc 3 as most relevant:
  'Neural networks learn patterns.'
```

### Key Observations

🎯 **Perfect Agreement on Top Result**: Both methods correctly identified **Document 3** ("Neural networks learn patterns.") as the most relevant match for the neural networks query.

📈 **Top 3 Rankings Match**: Interestingly, both approaches agreed on the top 3 most relevant documents:
1. **Doc 3**: "Neural networks learn patterns." (exact match)
2. **Doc 5**: "Deep learning uses neural networks." (related concept)  
3. **Doc 1**: "Machine learning is amazing." (broader field)

⚡ **Different Scoring Scales**: 
- **RandomForest**: Probabilistic outputs (0.150 = 15% confidence)
- **Cosine Similarity**: Geometric similarity (0.6648 = 66.48% similarity)

🔍 **Minor Ranking Differences**: While the top 3 matched, positions 4-5 differed slightly, showing how each method weighs different semantic relationships.
