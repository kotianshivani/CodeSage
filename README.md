# CodeSage: Software Engineering Exam Assistant
AI-powered study companion using pre-built textbook knowledge (RAG pipeline) to answer software engineering questions without textbook processing costs.

A Streamlit-based question-answering system that uses RAG (Retrieval-Augmented Generation) to answer questions about Software Engineering concepts.

## Features
- Pre-built ChromaDB vector store with software engineering textbook knowledge
- Zero data processing - Skip PDF parsing/chunking/embedding costs
- Groq LLM integration for cost-effective responses
- User-friendly Streamlit interface
- Semantic caching for faster responses
- Support for both OpenAI and Groq models
- Cache management with similarity threshold control
- Cache statistics tracking

## Prerequisites

- Python 3.8+
- OpenAI API key (for embeddings only) - https://platform.openai.com/api-keys
- Groq API key (free tier available) - https://console.groq.com/keys
OPENAI_API_KEY = "sk-your-openai-key"  # Only for embeddings
GROQ_API_KEY = "gsk-your-groq-key"    # Free tier available


## Setup and Installation
1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Add your API keys to Streamlit secrets
4. Run the app: `streamlit run app.py`
5. Ask questions through the web interface:
example - 
1. The probability of failure on demand (POFOD) for service from a system is 0.008 means that there is a __________ chance that a failure will occur when a demand is made.
Answer - 0.008
2. You can simply estimate the number of people required for a project team by dividing the total effort by the required project schedule. true or false?
Answer - False
3. Sociotechnical systems are so complex that it is impossible to understand them as a whole.
Answer - True

## Technical Architecture 
```
---
config:
  layout: fixed
  theme: dark
---
flowchart LR
    A["User Question"] --> B("Semantic Cache")
    B -- Cache Hit --> C["Instant Response"]
    B -- Cache Miss --> D["ChromaDB Vector Store"]
    D --> E["Document Store Lookup"]
    E --> F["Groq LLM Synthesis"]
    F --> G["Response"]

```

## Live Demo
https://codesage.streamlit.app/
