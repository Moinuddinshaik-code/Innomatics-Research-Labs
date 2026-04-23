# 🤖 RAG-Based Customer Support Assistant

A production-grade **Retrieval-Augmented Generation (RAG)** system with **LangGraph** workflow orchestration and **Human-in-the-Loop (HITL)** escalation for intelligent customer support.

## 🏗️ Architecture

```
User Query → Intent Classification → Document Retrieval → Confidence Evaluation
    ↓                                                            ↓
[Escalation?] ←──── YES ──── [Low Confidence?]          [High Confidence]
    ↓                                                            ↓
HITL Agent                                              LLM Response Generation
    ↓                                                            ↓
    └──────────────────→ Final Response ←────────────────────────┘
```

## 🛠️ Technology Stack

| Component | Technology |
|-----------|-----------|
| LLM | Groq (Llama 3.3 70B) |
| Embeddings | HuggingFace (Local) |
| Vector Store | ChromaDB |
| Workflow | LangGraph (StateGraph) |
| UI | Streamlit |
| PDF Processing | PyPDF + LangChain |

## 📦 Setup

```bash
# 1. Install dependencies
pip install -r requirements.txt
pip install reportlab  # For PDF generation only

# 2. Set up environment
cp .env.example .env
# Edit .env with your GOOGLE_API_KEY

# 3. Generate sample knowledge base
python generate_knowledge_base.py

# 4. Run the application
streamlit run app.py
```

## 🚀 Usage

### Web UI (Streamlit)
```bash
streamlit run app.py
```

### CLI
```bash
python cli.py
```

## 📁 Project Structure

```
├── app.py                      # Streamlit web application
├── cli.py                      # CLI interface
├── generate_knowledge_base.py  # Sample PDF generator
├── requirements.txt            # Dependencies
├── src/
│   ├── config.py               # Configuration
│   ├── document_processor.py   # PDF loading & chunking
│   ├── vector_store.py         # ChromaDB management
│   ├── retriever.py            # Document retrieval
│   ├── llm.py                  # LLM & prompts
│   ├── intent_classifier.py    # Intent detection
│   ├── graph_workflow.py       # LangGraph workflow
│   ├── hitl.py                 # HITL escalation
│   └── utils.py                # Utilities
├── docs/
│   ├── HLD.md                  # High-Level Design
│   ├── LLD.md                  # Low-Level Design
│   └── Technical_Documentation.md
├── knowledge_base/             # PDF documents
├── tests/                      # Test suite
└── chroma_db/                  # Vector store (auto-generated)
```

## 📄 Deliverables

1. **HLD** - `docs/HLD.md`
2. **LLD** - `docs/LLD.md`
3. **Technical Documentation** - `docs/Technical_Documentation.md`
4. **Working Project** - This codebase
