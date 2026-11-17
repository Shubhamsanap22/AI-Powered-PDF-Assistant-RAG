# 📄 AI-Powered PDF Assistant using RAG

An AI-powered PDF question-answering system built using **Retrieval-Augmented Generation (RAG)**. This application allows users to upload PDF documents and interact with their content through an AI-powered conversational chatbot.

The project uses **LangChain** for document processing and retrieval, **Mistral AI** for text embeddings and language generation, and **ChromaDB** as a vector database for storing and retrieving document embeddings.

The system retrieves relevant information from uploaded documents and generates context-grounded answers instead of relying entirely on the LLM's general knowledge.

## 🚀 Features

- **PDF Upload:** Upload PDF documents through an interactive Streamlit interface.
- **Document Processing:** Extract text from PDFs using PyPDFLoader.
- **Text Chunking:** Split documents into manageable chunks using RecursiveCharacterTextSplitter.
- **Text Embeddings:** Generate vector embeddings using Mistral AI's `mistral-embed` model.
- **Vector Database:** Store and retrieve document embeddings using ChromaDB.
- **Semantic Search:** Retrieve relevant document chunks using Maximum Marginal Relevance (MMR).
- **AI-Powered Responses:** Generate answers using Mistral Small.
- **Context-Grounded Answers:** Instruct the LLM to answer using only the retrieved document context.
- **Persistent Storage:** Reuse previously generated vector databases for the same uploaded PDF.
- **Conversational Interface:** Ask questions and interact with uploaded documents through Streamlit.

## 🛠️ Tech Stack

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| Streamlit | Web application interface |
| LangChain | RAG pipeline and orchestration |
| Mistral AI | Embeddings and LLM |
| ChromaDB | Vector database |
| PyPDF | PDF document loading |
| RecursiveCharacterTextSplitter | Text chunking |
| Python-dotenv | Environment variable management |

## 🧠 RAG Architecture

The application follows a Retrieval-Augmented Generation pipeline.

### 1. Document Ingestion

- Upload a PDF document.
- Extract text using PyPDFLoader.
- Split extracted text into smaller chunks.
- Generate embeddings using Mistral AI.
- Store embeddings in ChromaDB.

### 2. Retrieval

- Accept a user's question.
- Retrieve relevant document chunks using MMR search.
- Combine the retrieved chunks into contextual information.
- Pass the context and question to the LLM.

### 3. Generation

- Generate a prompt using the retrieved context and user's question.
- Send the prompt to Mistral Small.
- Generate a response grounded in the retrieved document content.

If the answer is unavailable in the retrieved context, the chatbot is instructed to respond:

> I could not find the answer in the document.

## 📂 Project Structure

```text
AI-Powered-PDF-Assistant-RAG/
│
├── app.py                # Streamlit PDF chatbot application
├── create_database.py   # PDF loading, chunking and embedding creation
├── main.py              # Command-line RAG chatbot
├── requirements.txt     # Python dependencies
├── .env                 # Mistral API key (not committed)
├── .gitignore           # Files excluded from Git
│
└── chroma_db/           # Persistent vector database
```

## ⚙️ Installation and Setup

### 1. Clone the Repository

```bash
git clone https://github.com/Shubhamsanap22/AI-Powered-PDF-Assistant-RAG.git
```

### 2. Navigate to the Project Directory

```bash
cd AI-Powered-PDF-Assistant-RAG
```

### 3. Create a Virtual Environment

```bash
python -m venv venv
```

Activate the environment:

**Windows:**
```bash
venv\Scripts\activate
```

**macOS / Linux:**
```bash
source venv/bin/activate
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

### 5. Configure Mistral API Key

Create a `.env` file in the root directory:

```env
MISTRAL_API_KEY=your_mistral_api_key
```

Get your API key from:

https://console.mistral.ai/

### 6. Run the Application

Start the Streamlit application:

```bash
streamlit run app.py
```

Open the local URL displayed in your terminal, upload a PDF document, and start asking questions!

## 🔍 Retrieval Configuration

The application uses Maximum Marginal Relevance (MMR) to retrieve relevant document chunks.

**Retrieval configuration:**

```python
search_type="mmr"

search_kwargs={
    "k": 4,
    "fetch_k": 10,
    "lambda_mult": 0.5
}
```

**Text splitting configuration:**

```python
chunk_size=1000
chunk_overlap=200
```

This configuration retrieves up to four relevant chunks while balancing relevance and diversity.

## 🤖 Models Used

| Component | Model |
|---|---|
| Embedding Model | mistral-embed |
| Language Model | mistral-small-2506 |
| Vector Database | ChromaDB |

## 🔐 Environment Variables

| Variable | Description |
|---|---|
| MISTRAL_API_KEY | API key required to access Mistral AI models |

**Important:** Never upload your `.env` file or API credentials to GitHub.

## 🔮 Future Improvements

- Support multiple document formats such as DOCX and TXT.
- Enable conversations across multiple PDF documents.
- Display source document references and page numbers.
- Add chat history persistence.
- Improve retrieval using reranking techniques.
- Support additional LLM providers.
- Deploy the application to a cloud platform.

## 👨‍💻 Author

**Shubham Sanap**

GitHub: [@Shubhamsanap22](https://github.com/Shubhamsanap22)

## ⭐ Show Your Support

If you find this project useful, consider giving it a ⭐ on GitHub!

---

*Built with Python, LangChain, Mistral AI, ChromaDB, and Streamlit.*
