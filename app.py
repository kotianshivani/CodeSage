import streamlit as st
from langchain_community.vectorstores import Chroma
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI
import uuid
from langchain.storage import InMemoryStore
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from base64 import b64decode
from dotenv import load_dotenv
import os
from langchain_groq import ChatGroq
from langchain.vectorstores import FAISS
from langchain.docstore.document import Document
import pickle
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')


# Initialize session state variables
if 'cache_hits' not in st.session_state:
    st.session_state.cache_hits = 0
if 'cache_misses' not in st.session_state:
    st.session_state.cache_misses = 0

# Load environment variables
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

# Function to parse documents (images and text)
def parse_docs(docs):
    b64 = []
    text = []
    for doc in docs:
        try:
            b64decode(doc)
            b64.append(doc)
        except Exception:
            text.append(doc)
    return {"images": b64, "texts": text}

# Updated function to build the prompt
def build_prompt(kwargs):
    docs_by_type = kwargs["context"]
    user_question = kwargs["question"]

    context_text = ""
    if len(docs_by_type["texts"]) > 0:
        for text_element in docs_by_type["texts"]:
            context_text += text_element.text

    prompt_template = """
    Answer the question based only on the following context, which can include text, tables, and the below image.
    Context: {context}
    Question: {question}
    """
    return ChatPromptTemplate.from_template(prompt_template).format(
        context=context_text,
        question=user_question
    )

# Initialize retriever
@st.cache_resource
def initialize_retriever():
    vectorstore = Chroma(
        collection_name="multi_modal_rag",
        embedding_function=OpenAIEmbeddings(),
        persist_directory="vectorstore_full_SE_RAG/"
    )
    import pickle
    with open("docstore_full.pkl", "rb") as f:
        docstore = pickle.load(f)

    retriever = MultiVectorRetriever(
        vectorstore=vectorstore,
        docstore=docstore,
        id_key="doc_id",
    )
    return retriever

# Initialize LLM pipeline
retriever = initialize_retriever()
chain = (
    {
        "context": retriever | RunnableLambda(parse_docs),
        "question": RunnablePassthrough(),
    }
    | RunnableLambda(build_prompt)
    | ChatGroq(
        model_name="llama-3.1-8b-instant",  # or "llama2-7b-32k" or other Groq models
        temperature=0.7,
        groq_api_key=os.getenv("GROQ_API_KEY")  # Make sure to set this in your .env file
    )
    | StrOutputParser()
)

class SemanticCache:
    def __init__(self):
        """Initialize semantic cache with OpenAI embeddings"""
        self.embedding_model = OpenAIEmbeddings()
        self.cache_file = "semantic_cache.pkl"
        self.vectorstore = self._initialize_cache()
    
    def _save_cache(self):
        """Save the current state of the cache"""
        try:
            # Save only the index and metadata
            index = self.vectorstore.index
            docstore = self.vectorstore.docstore
            index_to_docstore_id = self.vectorstore.index_to_docstore_id

            
            cache_data = {
                'index': index,
                'docstore': docstore,
                'index_to_docstore_id': index_to_docstore_id
            }
            
            with open(self.cache_file, "wb") as f:
                pickle.dump(cache_data, f)
        except Exception as e:
            st.error(f"Cache storage error: {str(e)}")
    
    def _initialize_cache(self):
        """Initialize or load existing FAISS vectorstore"""
        try:
            # Check if file exists and is not empty
            if os.path.exists(self.cache_file) and os.path.getsize(self.cache_file) > 0:
                with open(self.cache_file, "rb") as f:
                    cache_data = pickle.load(f)
                    
                    # Verify cache data structure
                    if isinstance(cache_data, dict) and 'index' in cache_data and 'docstore' in cache_data:
                        # Reconstruct the FAISS vectorstore
                        vectorstore = FAISS(
                            self.embedding_model,
                            index=cache_data['index'],
                            docstore=cache_data['docstore'],
                            index_to_docstore_id=cache_data['index_to_docstore_id']
                        )
                        return vectorstore
                    else:
                        st.warning("Invalid cache format, creating new cache")
            
            # Create new vectorstore if no valid cache exists
            st.info("Initializing new cache")
            return FAISS.from_texts(
                ["placeholder"], 
                self.embedding_model,
                metadatas=[{"question": "placeholder", "answer": "placeholder"}]
            )
            
        except (FileNotFoundError, EOFError, pickle.UnpicklingError) as e:
            st.warning(f"Cache file error: {str(e)}. Creating new cache.")
            # Delete corrupted cache file if it exists
            if os.path.exists(self.cache_file):
                os.remove(self.cache_file)
            
            # Return a new vectorstore
            return FAISS.from_texts(
                ["placeholder"], 
                self.embedding_model,
                metadatas=[{"question": "placeholder", "answer": "placeholder"}]
            )
        except Exception as e:
            st.error(f"Unexpected error during cache initialization: {str(e)}")
            return FAISS.from_texts(
                ["placeholder"], 
                self.embedding_model,
                metadatas=[{"question": "placeholder", "answer": "placeholder"}]
            )
    
    def get_response(self, question, similarity_threshold=0.85):
        """Get cached response if similar question exists"""
        try:
            results = self.vectorstore.similarity_search_with_score(
                question,
                k=1
            )
            
            if results and len(results) > 0:
                doc, score = results[0]
                similarity_score = 1 - score  # Convert distance to similarity
                
                if similarity_score >= similarity_threshold:
                    st.session_state.cache_hits += 1
                    return doc.metadata.get("answer"), True, similarity_score
            
            st.session_state.cache_misses += 1
            return None, False, 0
            
        except Exception as e:
            st.error(f"Cache lookup error: {e}")
            return None, False, 0
    
    def add_response(self, question, answer):
        """Add new question-answer pair to cache"""
        try:
            self.vectorstore.add_texts(
                [question],
                metadatas=[{"question": question, "answer": answer}]
            )
            self._save_cache()
        except Exception as e:
            st.error(f"Cache storage error: {e}")
    
    def clear_cache(self):
        """Clear the entire cache"""
        if os.path.exists(self.cache_file):
            os.remove(self.cache_file)
        self.vectorstore = self._initialize_cache()
        st.session_state.cache_hits = 0
        st.session_state.cache_misses = 0

# Initialize semantic cache
@st.cache_resource
def get_semantic_cache():
    return SemanticCache()

# Modified chain invocation with semantic caching
def answer_question_with_cache(question):
    cache = get_semantic_cache()
    
    # Check semantic cache
    cached_response, is_cached, similarity_score = cache.get_response(question)
    
    if is_cached:
        return cached_response, True, similarity_score
    
    # If no cache hit, use the chain
    response = chain.invoke(question)
    
    # Cache the new response
    cache.add_response(question, response)
    
    return response, False, 0

# Streamlit UI
st.title("Software Engineering Question Answering Bot")
st.write("Ask me anything based on the **Software Engineering** textbook!")

# Cache management in sidebar
with st.sidebar:
    st.title("Cache Management")
    
    # Display cache statistics
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Cache Hits", st.session_state.cache_hits)
    with col2:
        st.metric("Cache Misses", st.session_state.cache_misses)
    
    # Cache controls
    if st.button("Clear Cache"):
        cache = get_semantic_cache()
        cache.clear_cache()
        st.success("Cache cleared!")
        st.rerun()

    # Add similarity threshold slider
    similarity_threshold = st.slider(
        "Similarity Threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.85,
        help="Minimum similarity score required for cache hits"
    )

# Main question input
user_question = st.text_input("Enter your question:")

if user_question:
    with st.spinner("Fetching the answer..."):
        try:
            response, is_cached, similarity_score = answer_question_with_cache(user_question)
            
            # Show source of answer
            if is_cached:
                st.info(f"✨ Retrieved from cache (Similarity: {similarity_score:.2%})")
            else:
                st.info("🌐 Generated new response")
            
            st.success(f"Answer: {response}")
            
        except Exception as e:
            st.error(f"An error occurred: {e}")

# Optional: Force refresh button for cached answers
if user_question:
    if st.button("Get Fresh Answer"):
        with st.spinner("Generating new answer..."):
            try:
                response = chain.invoke(user_question)
                cache = get_semantic_cache()
                cache.add_response(user_question, response)
                st.info("🌐 Fresh answer generated")
                st.success(f"New Answer: {response}")
            except Exception as e:
                st.error(f"An error occurred: {e}")

st.sidebar.title("Instructions")
st.sidebar.write("""
1. Type a question related to the Software Engineering textbook.
2. Wait for the RAG pipeline to fetch the most relevant answer.
3. If there's an issue or the question is irrelevant, you'll get a clear response.
""")
