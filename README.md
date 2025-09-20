# ML Retrieval System

A machine learning-based document retrieval system with multiple approaches using Sentence Transformers.

## Features

- Document encoding using SentenceTransformers (all-MiniLM-L6-v2 model)
- **Two retrieval methods:**
  - RandomForest classifier for document similarity prediction with probability scoring
  - Direct cosine similarity for geometric similarity measurement
- Probability scoring and percentile ranking for query results
- Comprehensive comparison between different approaches
- Easy-to-use Python implementation

## Requirements

- Python 3.10+
- scikit-learn
- sentence-transformers
- numpy
- scipy

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
1. **Document Encoding**: Documents are converted to high-dimensional embeddings using the `all-MiniLM-L6-v2` model
2. **Classification Training**: A RandomForest classifier learns to map embeddings to document classes
3. **Query Processing**: Query text is encoded and classified to find the most relevant documents
4. **Ranking**: Results are ranked by probability and converted to percentiles for better interpretability

### Cosine Similarity Approach (cosine_similarity_retrieval.py)
1. **Document Encoding**: Documents are converted to embeddings using SentenceTransformers
2. **Similarity Calculation**: Direct cosine similarity is calculated between query and document embeddings
3. **Ranking**: Documents are ranked by similarity scores (0 = orthogonal, 1 = identical)
4. **Results**: More intuitive and interpretable similarity scores

### Method Comparison
- **RandomForest**: Uses machine learning to capture complex patterns, provides probabilistic outputs
- **Cosine Similarity**: Direct geometric measurement, faster computation, more interpretable results

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

### Cosine Similarity Approach
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