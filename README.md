# LangChain RAG Demo with Streamlit and Gemini API

This project demonstrates a simple Retrieval-Augmented Generation (RAG) system using LangChain, Streamlit for the UI, and the Google Gemini API for language model capabilities. ChromaDB is used as the vector store.

## Features

- **Streamlit UI**: An interactive web interface to upload PDF documents and ask questions.
- **LangChain**: Orchestrates the RAG pipeline, including document loading, text splitting, embedding, and conversational chain.
- **Google Gemini API**: Powers the embeddings and the conversational AI model.
- **ChromaDB**: A lightweight and easy-to-set-up vector database for storing document embeddings.

## Setup and Installation

Follow these steps to set up and run the project locally:

### 1. Clone the repository (if you haven't already)

```bash
git clone <repository_url>
cd lang-chain-rag-demo
```

### 2. Create a Python Virtual Environment (Recommended)

```bash
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
```

### 3. Install Dependencies

Install the required Python packages using `pip`:

```bash
pip install -r requirements.txt
```

### 4. Set up your Google API Key

1. Obtain a Google API Key from the [Google AI Studio](https://aistudio.google.com/app/apikey).
2. Create a `.env` file in the root directory of the project (where `app.py` is located) and add your API key:

   ```
   GOOGLE_API_KEY="YOUR_API_KEY"
   ```
   Replace `YOUR_API_KEY` with your actual Google API Key.

### 5. Run the Streamlit Application

```bash
streamlit run app.py
```

This will open the Streamlit application in your web browser. If it doesn't open automatically, navigate to `http://localhost:8501`.

## How to Use

1. **Upload PDF Documents**: On the sidebar, use the "Upload your PDF Documents" section to upload one or more PDF files. These documents will be used to populate the RAG system.
2. **Process Documents**: Click the "Process Documents" button after uploading your PDFs. This will extract text, split it into chunks, create embeddings, and store them in ChromaDB.
3. **Ask Questions**: Once the documents are processed, you can type your questions in the "Your Question:" input box. The RAG system will retrieve relevant information from your uploaded documents and generate an answer using the Gemini API.

## Project Structure

```
lang-chain-rag-demo/
├── .env                 # Stores your Google API Key
├── app.py               # Main Streamlit application with RAG logic
├── requirements.txt     # Python dependencies
└── README.md            # Project setup and usage instructions
```

## Data Population for RAG

The RAG system is populated with data by uploading PDF documents through the Streamlit UI. When you upload PDFs and click "Process Documents", the `app.py` script performs the following:

1.  **Loads PDF Text**: Uses `PyPDFLoader` to extract text from the uploaded PDF files.
2.  **Splits Text into Chunks**: Divides the extracted text into smaller, manageable chunks using `RecursiveCharacterTextSplitter`. This is crucial for efficient retrieval.
3.  **Creates Embeddings**: Generates vector embeddings for each text chunk using `GoogleGenerativeAIEmbeddings` (powered by the Gemini Pro model).
4.  **Stores in Vector Store**: Stores these embeddings along with the original text chunks in ChromaDB. The vector store is persisted locally in a directory named `./chroma_db`.

This process effectively "populates" the RAG system with your document data, making it ready to answer questions based on that content.