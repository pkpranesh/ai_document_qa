import streamlit as st
from google import genai
import PyPDF2
import chromadb
from sentence_transformers import SentenceTransformer
import os

st.set_page_config(
    page_title="AI Document Q&A",
    page_icon="🤖",
    layout="wide"
)

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY", "YOUR_KEY"))

embedder = SentenceTransformer('all-MiniLM-L6-v2')

def extract_text(pdf_file):
    reader = PyPDF2.PdfReader(pdf_file)
    text   = " ".join(page.extract_text() 
                      for page in reader.pages)
    return text

def split_into_chunks(text, chunk_size=500, overlap=50):
    chunks = []
    start  = 0
    
    while start < len(text):
        end   = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start = end - overlap  # overlap with next chunk
    
    return chunks

def create_vector_store(chunks):
    # Create ChromaDB client
    chroma_client = chromadb.Client()
    
    # Delete existing collection if exists
    try:
        chroma_client.delete_collection("documents")
    except:
        pass
    
    # Create fresh collection
    collection = chroma_client.create_collection("documents")
    
    # Embed all chunks and store
    embeddings = embedder.encode(chunks).tolist()
    
    collection.add(
        documents  = chunks,
        embeddings = embeddings,
        ids        = [f"chunk_{i}" for i in range(len(chunks))]
    )
    
    return collection

def find_relevant_chunks(question, collection, top_k=3):
    # Embed the question
    question_embedding = embedder.encode([question]).tolist()
    
    # Search for similar chunks
    results = collection.query(
        query_embeddings = question_embedding,
        n_results        = top_k
    )
    
    return results['documents'][0]  # top 3 chunks

def answer_question(question, relevant_chunks, client):
    # Combine relevant chunks into context
    context = "\n\n".join(relevant_chunks)
    
    prompt = f"""
    Answer the question based ONLY on the context below.
    If the answer is not in the context, say "I couldn't find 
    this information in the document."
    
    Context:
    {context}
    
    Question: {question}
    
    Answer:
    """
    
    response = client.models.generate_content(
        model    = 'gemini-3.8-flash',
        contents = prompt
    )
    return response.text

# ── UI ────────────────────────────────────────────────────
st.title("🤖 AI Document Q&A")
st.markdown("### Chat with any PDF document using AI!")
st.markdown("---")

# ── File Upload ───────────────────────────────────────────
uploaded_file = st.file_uploader(
    "Upload a PDF document",
    type=['pdf']
)

if uploaded_file:
    with st.spinner("📄 Processing document..."):
        # Extract text
        text   = extract_text(uploaded_file)
        
        # Split into chunks
        chunks = split_into_chunks(text)
        
        # Create vector store
        collection = create_vector_store(chunks)
    
    st.success(f"✅ Document processed! "
               f"Created {len(chunks)} chunks.")
    
    st.markdown("---")
    
    # ── Chat Interface ─────────────────────────────────────
    st.markdown("### 💬 Ask Questions About Your Document")
    
    # Store chat history
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    
    # Display chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg['role']):
            st.markdown(msg['content'])
    
    # Chat input
    question = st.chat_input("Ask anything about the document...")
    
    if question:
        # Add user message
        st.session_state.messages.append({
            'role'   : 'user',
            'content': question
        })
        with st.chat_message('user'):
            st.markdown(question)
        
        # Get answer
        with st.spinner("🤔 Thinking..."):
            relevant = find_relevant_chunks(question, collection)
            answer   = answer_question(question, relevant, client)
        
        # Add assistant message
        st.session_state.messages.append({
            'role'   : 'assistant',
            'content': answer
        })
        with st.chat_message('assistant'):
            st.markdown(answer)

else:
    st.info("👆 Upload a PDF to start chatting!")

# ── Sidebar ───────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 📖 How It Works")
    st.markdown("""
    1. Upload any PDF
    2. Document split into chunks
    3. Chunks converted to vectors
    4. Ask any question
    5. AI finds relevant chunks
    6. Gemini answers from document
    """)
    
    if 'messages' in st.session_state:
        if st.button("🗑️ Clear Chat"):
            st.session_state.messages = []
            st.rerun()
    
    st.markdown("---")
    st.markdown("**Built by Pranesh P K**")
    st.markdown("SRM IST | B.Tech ECE | 2026")

# ── Footer ────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<p style='text-align:center; color:gray;'>"
    "AI Document Q&A | RAG System | "
    "Powered by Gemini + ChromaDB | "
    "Built by Pranesh P K"
    "</p>",
    unsafe_allow_html=True
)