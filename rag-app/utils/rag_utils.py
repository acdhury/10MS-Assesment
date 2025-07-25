import ollama
import re
from langdetect import detect
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.memory import ConversationBufferMemory
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

# Initialize the RAG system components
def initialize_rag_system():
    # Load and process the Bangla text
    with open('rag-app/bangla-text.txt', 'r', encoding='utf-8') as f:
        text = f.read()
    
    processed_text = process_bangla_text(text)
    
    # Text splitting
    bangla_splitter = RecursiveCharacterTextSplitter(
        chunk_size=80,
        chunk_overlap=10,
        separators=["\n\n", "\n", "।", "?", "!", ",", " ", ""]
    )
    chunks = bangla_splitter.split_text(processed_text)
    
    # Create vector database
    vector_db = create_vector_db(chunks)
    
    # Initialize memory
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    
    return {
        'vector_db': vector_db,
        'memory': memory
    }

# Text processing functions
def process_bangla_text(text):
    text = re.sub(
        r'[^\u0980-\u09FF\u09BC\u09BE-\u09CC\u0020-\u007E\u0964\u0965]', 
        ' ', 
        text
    )
    text = re.sub(r'\s+', ' ', text).strip()
    text = re.sub(r'([ক-হ]্)\s+([ক-হ])', r'\1\2', text)
    text = re.sub(r'([অ-ঔ])([া-ৌ])', r'\1\2', text)
    return text

def create_vector_db(text_chunks):
    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-m3",
                                     encode_kwargs={'normalize_embeddings': True})
    vector_db = FAISS.from_texts(text_chunks, embeddings)
    return vector_db

def detect_and_preprocess(query):
    lang = detect(query)
    if lang == 'bn':
        query = re.sub(r'[^\u0980-\u09FF\u0020-\u007E\u0964\u0965]', ' ', query)
    else:
        query = re.sub(r'[^\w\s]', ' ', query)
    return query.strip(), lang

def retrieve_context(query, vector_db, lang, top_k=5, score_threshold=0.6):
    query_clean = query.strip()
    docs_with_scores = vector_db.similarity_search_with_relevance_scores(
        query_clean,
        k=top_k
    )
    
    filtered = []
    for doc, score in docs_with_scores:
        if score >= score_threshold:
            filtered.append(f"[Confidence: {score:.2f}] {doc.page_content}")
    
    return filtered if filtered else ["No relevant information found"], lang

def generate_response(query, vector_db, memory):
    client = ollama.Client()
    query_clean, lang = detect_and_preprocess(query)
    context, lang = retrieve_context(query_clean, vector_db, lang)

    if lang == 'bn':
        prompt = f"""
        তথ্য:{' '.join(context)}
        আলাপের ইতিহাস:{memory.load_memory_variables({})["chat_history"]}
        প্রশ্ন: {query}
        প্রাসঙ্গিক তথ্য ব্যবহার করে স্পষ্ট, সংক্ষিপ্ত এবং সঠিক উত্তর দিন। নিজে থেকে কিছু লিখবেন না। যদি আপনি প্রাসঙ্গিক তথ্য খুঁজে না পান তবে বলুন যে আপনার কাছে তা নেই।"""
    else:
        prompt = f"""
        Information:{' '.join(context)}
        Conversation History:{memory.load_memory_variables({})["chat_history"]}
        Question: {query}
        Provide a clear concise and accurate answer using the relevant information. Do not write anything on your own. If you can not find relevant information say you don't have them."""

    response = client.generate(model='deepseek-r1:7b', prompt=prompt)
    memory.save_context({"question": query}, {"answer": response['response']})
    return response['response']