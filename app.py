import streamlit as st
import os
import pathlib
import textwrap
from PIL import Image
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Import necessary components from LangChain and Google Generative AI
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from tools.web_search import web_search


# For displaying Markdown in Streamlit (though not directly used in the final Streamlit UI, good for debugging/testing)


# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()
GOOGLE_API_KEY=os.environ["GOOGLE_API_KEY"]

# Configure the Google Generative AI with the API key
import google.generativeai as genai
if GOOGLE_API_KEY:
    genai.configure(api_key=GOOGLE_API_KEY)
else:
    st.warning("Please enter your Google API Key!")
    logging.warning("Google API Key not set.")


import tempfile

# Function to extract text from PDF documents
def get_pdf_text(pdf_docs):
    logging.info("Starting PDF text extraction.")
    text = ""
    for pdf in pdf_docs:
        # Save the uploaded file to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(pdf.read())
            tmp_file_path = tmp_file.name

        # Initialize PyPDFLoader with the temporary file path
        pdf_reader = PyPDFLoader(tmp_file_path)
        # Extract text from each Document and append as a single string
        documents = pdf_reader.load_and_split()
        text += "\n".join([doc.page_content for doc in documents])
        logging.info(f"Extracted text from {tmp_file.name}.")

    logging.info("Finished PDF text extraction.")
    return text

# Function to split text into smaller, manageable chunks
def get_text_chunks(text):
    logging.info("Starting text chunking.")
    # Initialize RecursiveCharacterTextSplitter with specified chunk size and overlap
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=10000,
        chunk_overlap=1000
    )
    # Split the input text into chunks
    chunks = text_splitter.split_text(text)
    logging.info(f"Created {len(chunks)} text chunks.")
    return chunks

# Function to create a vector store from text chunks
def get_vector_store(text_chunks):
    logging.info("Starting vector store creation.")
    # Initialize GoogleGenerativeAIEmbeddings for creating embeddings
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    # Create a Chroma vector store from the text chunks and embeddings
    # The vector store will be persisted to the './chroma_db' directory
    vector_store = Chroma.from_texts(text_chunks, embedding=embeddings, persist_directory="./chroma_db")
    # Persist the vector store to disk
    vector_store.persist()
    logging.info("Vector store created and persisted.")

# Function to set up the conversational chain for Q&A
def get_conversational_chain():
    logging.info("Setting up conversational chain.")
    # Define the prompt template for the question-answering model
    prompt_template = """Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context, just say, "answer is not available in the context", don't try to make up an answer.

    Context:
    {context}

    Chat History:
    {chat_history}

    Question:
    {question}

    Answer:
    """

    # Initialize the ChatGoogleGenerativeAI model
    model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.3)
    # Create a PromptTemplate with the defined template and input variables
    prompt = PromptTemplate(
        template=prompt_template, input_variables=["context", "question", "chat_history"]
    )
    # Load the question-answering chain with the model and prompt
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    logging.info("Conversational chain set up.")
    return chain

# Function to handle user input and generate a response
def user_input(user_question, chat_history):
    logging.info(f"User question received: {user_question}")
    # Initialize embeddings for searching the vector store
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    # Load the persisted Chroma vector store
    new_db = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
    # Perform a similarity search to retrieve relevant documents based on the user's question
    docs = new_db.similarity_search(user_question)
    logging.info(f"Found {len(docs)} relevant documents for the question.")

    # Get the conversational chain
    chain = get_conversational_chain()

    # Get the response from the conversational chain
    response = chain(
        {"input_documents": docs, "question": user_question, "chat_history": chat_history}, return_only_outputs=True
    )
    response_text = response["output_text"]
    logging.info(f"Response generated: {response_text}")

    # Check if the response indicates that the answer is not in the documents
    # This is a heuristic and might need refinement based on model behavior
    if "I don't know" in response_text or "answer is not available in the context" in response_text or "based on the information provided" in response_text:
            logging.info("Answer not found in documents, performing web search.")
            web_search_results = web_search(query=user_question)
            if web_search_results:
                # You might want to process these results further or summarize them
                # For now, just return the first few results as a string
                search_summary = "\n\nWeb Search Results:\n"
                for i, result in enumerate(web_search_results[:3]): # Limit to top 3 results
                    search_summary += f"{i+1}. {result['title']} - {result['link']}\n  {result['snippet']}\n\n"
                return response_text + search_summary
            else:
                return response_text + "\n\nNo relevant web search results found."
    else:
        return response_text


def main():
    logging.info("Starting Streamlit application.")
    # Set Streamlit page configuration
    st.set_page_config(page_title="Gemini RAG Chatbot")
    # Display the main title of the application
    st.title("Gemini RAG Chatbot")

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Sidebar for API Key input and document processing
    with st.sidebar:
        st.header("Configuration")
        # Input field for Google API Key
        if not GOOGLE_API_KEY:
            google_api_key = st.text_input("Google API Key", type="password", key="google_api_key_input")
        # Check if API key is provided and set it as an environment variable
            if google_api_key:
                os.environ["GOOGLE_API_KEY"] = google_api_key
                st.success("API Key set!")
                logging.info("Google API Key set from sidebar.")
            else:
                st.warning("Please enter your Google API Key!")
                logging.warning("Google API Key not set.")

        st.subheader("Upload your PDF Documents")
        # File uploader for PDF documents
        pdf_docs = st.file_uploader(
            "Upload your PDF Files and Click on the Process Button",
            accept_multiple_files=True,
        )
        # Button to trigger document processing
        if st.button("Process Documents"):
            if pdf_docs:
                with st.spinner("Processing..."):
                    # Get raw text from uploaded PDFs
                    raw_text = get_pdf_text(pdf_docs)
                    # Get text chunks from the raw text
                    text_chunks = get_text_chunks(raw_text)
                    # Create and persist the vector store
                    get_vector_store(text_chunks)
                    st.success("Done")
                    logging.info("Document processing completed successfully.")
            else:
                st.warning("Please upload PDF documents to process.")
                logging.warning("Process Documents button clicked without PDFs.")

    # Main area for user interaction
    # Input field for user's question
    if prompt := st.chat_input("Ask me anything about the uploaded documents!"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                # Prepare chat history for the model
                # Exclude the current user prompt from chat history passed to the model
                chat_history_str = "\n".join([f"{msg['role']}: {msg['content']}" for msg in st.session_state.messages[:-1]])
                response = user_input(prompt, chat_history_str)
                st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})

# Entry point of the application
if __name__ == "__main__":
    main()