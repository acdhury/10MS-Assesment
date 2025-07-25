# For checking the rag model from IPYNB format

## Implementation Details

### Text Extraction
- Used tesseract-ocr for extracting Bangla texts from the given PDF using the OCR method
- Compiled all the text in a text file 
- Used the text file as the primary knowledge base source

### Text Processing
- Uses regex patterns specifically designed for Bangla text cleaning
- Handles special cases for Bangla compound characters
- Preserves Bangla punctuation (।, ?, !)

### Chunking Approach
- RecursiveCharacterTextSplitter configured with:
  - Chunk size: 200 characters
  - Overlap: 30 characters
  - Separators: ["\n\n", "\n", "।", "?", "!", ",", " "]

### Embeddings and Retrieval
- Uses HuggingFace's "BAAI/bge-m3" embedding model
- FAISS for vector storage and similarity search
- Cosine similarity with a threshold of 0.5

### Current Limitations
1. No API implementation (runs only in notebook)
2. No formal evaluation metrics
3. Basic chunking without semantic awareness
4. No query expansion capabilities

## How to Use
1. Run cells sequentially in the notebook
2. For queries:
   ```python
   query = "Your Bangla query here"
   similar_docs = vector_db.similarity_search_with_relevance_scores(query)
   print(similar_docs)

   ```

# For accessing the web-app
- Go to the rag-app directory using the command line
```bash
  cd rag-app
```

Install the additional dependencies
```bash
pip install -r requirements.txt
```

- run the app.py
```bash
python app.py
```
