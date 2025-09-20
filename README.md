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

## 📈 Scalability Analysis & Real-World Impact

### Current Implementation Scale
- **Document Set**: 10 sample documents (proof-of-concept)
- **Embedding Dimension**: 384 (all-MiniLM-L6-v2)
- **Query Processing**: Real-time for small collections

### Scalability Potential

#### **Cosine Similarity Approach** ⭐ **Recommended for Scale**
- **Linear Scalability**: O(n) complexity for n documents
- **Memory Efficient**: Can use approximate similarity search (FAISS, Annoy)
- **Real-time Queries**: Sub-second response for millions of documents
- **Dynamic Updates**: Easy addition/removal of documents
- **Production Ready**: Used by major search engines and recommendation systems

**Scaling Strategies**:
```
Small Scale (1K-10K docs)     → Direct in-memory cosine similarity
Medium Scale (10K-1M docs)    → FAISS with IVF indexing
Large Scale (1M+ docs)        → Distributed FAISS with sharding
Enterprise Scale (10M+ docs)  → Vector databases (Pinecone, Weaviate, Qdrant)
```

#### **RandomForest Approach** ⚠️ **Limited Scalability**
- **Non-scalable**: Training complexity grows with document count
- **Memory Intensive**: Must store entire model for all document classes
- **Static Collections**: Requires retraining for new documents
- **Academic Interest**: Useful for research but not production systems

### Real-World Applications & Impact

#### **Immediate Applications** (with cosine similarity)
1. **Enterprise Search**: Internal document retrieval systems
2. **Customer Support**: Automated ticket routing and knowledge base search
3. **Content Recommendation**: Blog posts, articles, product recommendations
4. **Research Tools**: Academic paper discovery and citation networks
5. **FAQ Systems**: Automated question-answer matching

#### **Production Deployment Scenarios**
```python
# Small Company (1K-10K documents)
# - Direct implementation, minimal infrastructure
# - Cost: Low, Response Time: <100ms

# Medium Company (100K documents)  
# - FAISS indexing, single server deployment
# - Cost: Moderate, Response Time: <50ms

# Large Enterprise (1M+ documents)
# - Distributed vector database, cloud deployment  
# - Cost: High, Response Time: <10ms, High availability
```

#### **Industry Impact Potential**
- **Search Industry**: Alternative to traditional keyword-based search
- **E-commerce**: Semantic product search ("comfortable running shoes" → actual comfort-focused products)
- **Legal Tech**: Case law similarity and precedent finding
- **Healthcare**: Medical literature retrieval and diagnostic support
- **Education**: Personalized learning content recommendation

### Performance Benchmarks (Projected)

| Scale | Documents | Index Time | Query Time | Memory Usage |
|-------|-----------|------------|------------|--------------|
| Small | 1K-10K | <1 min | <10ms | <100MB |
| Medium | 10K-100K | <10 min | <50ms | <1GB |
| Large | 100K-1M | <1 hour | <100ms | <10GB |
| Enterprise | 1M+ | <1 day | <10ms | <100GB |

*Note: Using approximate similarity search with 95%+ accuracy*

## Installation

1. Clone this repository
2. Install dependencies:
   ```bash
   pip install scikit-learn sentence-transformers numpy scipy
   ```

## Usage

### RandomForest Approach
Run the RandomForest-based retrieval system:
```bash
python main.py
```

### Cosine Similarity Approach
Run the cosine similarity-based retrieval system:
```bash
python cosine_similarity_retrieval.py
```

### Compare Both Methods
Run a side-by-side comparison of both approaches:
```bash
python comparison.py
```

Each system will:
1. Load a sample dataset of 10 documents
2. Encode them using SentenceTransformers
3. Process a query using the respective method
4. Return ranked results with scores

## How it works

### RandomForest Approach (main.py)
**The Problem**: This approach artificially treats document retrieval as a classification problem.

1. **Document Encoding**: Documents → high-dimensional embeddings (384D vectors)
2. **Artificial Classification**: Each document becomes a "class" (Doc 1, Doc 2, etc.)
3. **Model Training**: RandomForest learns to classify embeddings into document classes
4. **Query Processing**: Query embedding → predict which "document class" it belongs to
5. **Probabilistic Output**: Get confidence scores for all document classes

**Why This Is Interesting**: Tests whether ensemble methods can learn semantic relationships that pure similarity misses.

**Why This Is Problematic**: Documents aren't natural classes - this creates an artificial ML problem.

### Cosine Similarity Approach (cosine_similarity_retrieval.py) ⭐
**The Solution**: Direct mathematical similarity measurement in vector space.

1. **Document Encoding**: Documents → embeddings using pre-trained transformer
2. **Query Encoding**: Query → same embedding space  
3. **Similarity Calculation**: cos(θ) between query vector and each document vector
4. **Ranking**: Sort by similarity scores (higher = more similar)
5. **Interpretation**: Scores represent semantic closeness (0=unrelated, 1=identical)

**Why This Works**: Leverages the geometric properties of transformer embeddings where semantically similar texts cluster together in vector space.

### Mathematical Foundation
```
Cosine Similarity = (A · B) / (||A|| × ||B||)

Where:
- A = query embedding vector
- B = document embedding vector  
- Result ranges from -1 (opposite) to 1 (identical)
- Measures angle between vectors, ignoring magnitude
```

## Example Output

### RandomForest Approach
```
Query: 'I want to learn about neural networks.'

Predicted closest doc: Doc 3
Predicted document text: 'Neural networks learn patterns.'

Detailed Results:
Doc #  Probability  Percentile   Document Text
Doc 3  0.150        100.0        Neural networks learn patterns.
Doc 5  0.140        88.9         Deep learning uses neural networks.
...
```

### Cosine Similarity Approach ⭐
```
Query: 'I want to learn about neural networks.'

Most similar document: Doc 3
Document text: 'Neural networks learn patterns.'
Cosine similarity score: 0.6648

All Documents Ranked by Similarity:
Rank   Doc #  Similarity   Document Text
1      3      0.6648       Neural networks learn patterns.
2      5      0.5775       Deep learning uses neural networks.
...
```

### Comparison Results
```
Both methods agree on top result: YES
Both methods identified Doc 3 as most relevant:
  'Neural networks learn patterns.'

Method Characteristics:
┌─ RandomForest: Academic interest, not scalable
└─ Cosine Similarity: Production-ready, industry standard
```

## 🚀 Next Steps for Production

### Immediate Improvements
1. **Batch Processing**: Process multiple queries simultaneously
2. **Caching**: Store embeddings to avoid recomputation
3. **Indexing**: Implement FAISS for approximate similarity search
4. **API Wrapper**: RESTful API for integration with applications

### Scaling Architecture
```python
# Recommended Production Stack:
# 1. FastAPI for REST endpoints
# 2. FAISS for vector indexing  
# 3. Redis for embedding cache
# 4. PostgreSQL for metadata storage
# 5. Docker for containerization
```

### Research Extensions
- **Multi-modal Retrieval**: Text + image + metadata
- **Domain Adaptation**: Fine-tune embeddings for specific domains
- **Hybrid Search**: Combine semantic + keyword search
- **Evaluation Metrics**: Precision@K, NDCG, user satisfaction studies

---

*This project demonstrates the fundamental concepts behind modern semantic search systems used by companies like Google, OpenAI, and Pinecone. The cosine similarity approach represents the industry standard for vector-based retrieval systems.*