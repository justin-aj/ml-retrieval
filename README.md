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
