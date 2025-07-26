import streamlit as st
import os
import pathlib
import textwrap
from PIL import Image
import logging
import streamlit as st
from RealtimeTTS import TextToAudioStream, SystemEngine
from RealtimeSTT import AudioToTextRecorder
import nltk
nltk.data.path.append('./nltk_data')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Import necessary components from LangChain and Google Generative AI
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
# from tools.web_search import web_search


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

# Check for VAPI_API_KEY (re-adding this check as it was part of the previous integration)
# VAPI_API_KEY = os.environ.get("VAPI_API_KEY")
# if not VAPI_API_KEY:
#     logging.warning("VAPI_API_KEY environment variable not set.")
#     st.warning("VAPI_API_KEY environment variable not set. Vapi.ai features will not be available.")


import tempfile

# Function to extract text from PDF documents
def get_document_text(uploaded_file):
    logging.info(f"Starting text extraction for {uploaded_file.name}.")
    text = ""
    file_extension = os.path.splitext(uploaded_file.name)[1].lower()

    if file_extension == ".pdf":
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_file_path = tmp_file.name
        pdf_reader = PyPDFLoader(tmp_file_path)
        documents = pdf_reader.load_and_split()
        text = "\n".join([doc.page_content for doc in documents])
        os.unlink(tmp_file_path) # Clean up the temporary file
    else:
        logging.warning(f"Unsupported file type: {file_extension}")
        st.warning(f"Unsupported file type: {file_extension}. Only PDF files are supported for now.")
        return None

    logging.info(f"Finished text extraction for {uploaded_file.name}.")
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
def get_vector_store(text_chunks, file_name):
    logging.info(f"Starting vector store creation for {file_name}.")
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    # Add metadata to each chunk
    metadatas = [{
        "source": file_name,
        "upload_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }] * len(text_chunks)

    vector_store = Chroma.from_texts(text_chunks, embedding=embeddings, metadatas=metadatas, persist_directory="./chroma_db")
    vector_store.persist()
    logging.info(f"Vector store created and persisted for {file_name}.")

from datetime import datetime

def get_uploaded_files_from_vectordb():
    logging.info("Checking Chroma DB for uploaded files.")
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        # Load the persisted Chroma vector store
        db = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
        # Retrieve all documents to get their metadata
        # This might be inefficient for very large dbs, consider a custom Chroma retriever if performance is an issue
        docs = db.get(include=['metadatas'])
        if docs and 'metadatas' in docs:
            # Extract unique filenames from metadata
            unique_files = sorted(list(set([m['source'] for m in docs['metadatas'] if 'source' in m])))
            logging.info(f"Found {len(unique_files)} unique files in Chroma DB.")
            return unique_files
        else:
            logging.info("No documents or metadata found in Chroma DB.")
            return []
    except Exception as e:
        logging.error(f"Error accessing Chroma DB: {e}")
        return []

# Function to set up the conversational chain for Q&A
def get_conversational_chain():
    logging.info("Setting up conversational chain.")
    # Define the prompt template for the question-answering model
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Answer the user's questions based on the below context:\n\n{context}"),
        ("human", "{input}"),
    ])

    # Initialize the ChatGoogleGenerativeAI model
    model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.3, stream=True)
    # Create a stuff documents chain
    chain = create_stuff_documents_chain(model, prompt)
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
    full_response_content = ""
    for chunk in chain.stream(
        {"context": docs, "input": user_question}
    ):
        content_to_add = ""
        if hasattr(chunk, 'content'):
            content_to_add = chunk.content
        elif isinstance(chunk, dict):
            if "answer" in chunk:
                content_to_add = chunk["answer"]
            elif "page_content" in chunk:
                content_to_add = chunk["page_content"]
            elif "content" in chunk:
                content_to_add = chunk["content"]
        elif hasattr(chunk, 'page_content'): # For Document objects
            content_to_add = chunk.page_content
        else:
            content_to_add = str(chunk)
        full_response_content += content_to_add
        yield content_to_add

    logging.info(f"Full response generated: {full_response_content}")

    # Check if the response indicates that the answer is not in the documents
    # This is a heuristic and might need refinement based on model behavior
    # if "I don't know" in full_response_content or "answer is not available in the context" in full_response_content or "based on the information provided" in full_response_content:
    #         logging.info("Answer not found in documents, performing web search.")
    #         web_search_results = web_search(query=user_question)
    #          if web_search_results:
    #             # You might want to process these results further or summarize them
    #             # For now, just return the first few results as a string
    #             search_summary = "\n\nWeb Search Results:\n"
    #             for i, result in enumerate(web_search_results[:3]): # Limit to top 3 results
    #                 search_summary += f"{i+1}. {result['title']} - {result['link']}\n  {result['snippet']}\n\n"
    #             yield search_summary
    #         else:
    #             yield "\n\nNo relevant web search results found."




def main():
    logging.info("Starting Streamlit application.")
    # Set Streamlit page configuration
    st.set_page_config(page_title="Gemini RAG Voicebot")
    # Display the main title of the application
    st.title("Gemini RAG Chatbot")

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Add a section for voice interaction
    st.subheader("Voice Interaction")
    col1, col2 = st.columns([1, 4])
    with col1:
        record_button = st.button("🎤 Record")
    with col2:
        stop_button = st.button("⏹️ Stop")

    if 'recording' not in st.session_state:
        st.session_state.recording = False
    if 'audio_recorder' not in st.session_state:
        st.session_state.audio_recorder = None

    if record_button and not st.session_state.recording:
        st.session_state.recording = True
        st.session_state.audio_recorder = AudioToTextRecorder(spinner=False)
        st.session_state.audio_recorder.start()
        st.info("Recording... Speak now!")

    if stop_button and st.session_state.recording:
        st.session_state.recording = False
        if st.session_state.audio_recorder:
            st.session_state.audio_recorder.stop()
            transcribed_text = st.session_state.audio_recorder.text()
            st.session_state.audio_recorder = None
            if transcribed_text:
                st.session_state.messages.append({"role": "user", "content": transcribed_text})
                with st.chat_message("user"):
                    st.markdown(transcribed_text)
                
                # Generate response from RAG
                with st.spinner("Thinking..."):
                    response_container = st.empty()
                    full_response_content = ""
                    with st.chat_message("assistant"):
                        tts_stream = TextToAudioStream(engine=SystemEngine())
                        for chunk in user_input(transcribed_text, st.session_state.messages):
                            full_response_content += chunk
                            response_container.markdown(full_response_content)
                            try:
                                tts_stream.feed(chunk)
                            except Exception as e:
                                logging.error(f"Error feeding audio chunk: {e}")
                        try:
                            tts_stream.play()
                        except Exception as e:
                            logging.error(f"Error playing audio stream: {e}")
                    st.session_state.messages.append({"role": "assistant", "content": full_response_content})
        else:
                st.warning("No speech detected.")

    # Handle text input for chat
    if prompt := st.chat_input("Ask your question here..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.spinner("Thinking..."):
            response_container = st.empty()
            full_response_content = ""
            with st.chat_message("assistant"):
                tts_stream = TextToAudioStream(engine=SystemEngine())
                for chunk in user_input(prompt, st.session_state.messages):
                    full_response_content += chunk
                    response_container.markdown(full_response_content)
                    try:
                        tts_stream.feed(chunk)
                    except Exception as e:
                        logging.error(f"Error feeding audio chunk: {e}")
                try:
                    tts_stream.play()
                except Exception as e:
                    logging.error(f"Error playing audio stream: {e}")
            st.session_state.messages.append({"role": "assistant", "content": full_response_content})


    with st.sidebar:
        st.header("Configuration")
        # Input field for Google API Key
        if not GOOGLE_API_KEY:
            google_api_key = st.text_input("Google API Key", type="password", key="google_api_key_input")
            if google_api_key:
                os.environ["GOOGLE_API_KEY"] = google_api_key
                st.success("API Key set!")
                logging.info("Google API Key set from sidebar.")
            else:
                st.warning("Please enter your Google API Key!")
                logging.warning("Google API Key not set.")

        st.subheader("Upload Documents")
        uploaded_files = st.file_uploader(
            "Upload your document files (e.g., PDF)",
            accept_multiple_files=True,
            type=["pdf"]
        )

        if st.button("Process Uploaded Documents"):
            if uploaded_files:
                for uploaded_file in uploaded_files:
                    with st.spinner(f"Processing {uploaded_file.name}..."):
                        raw_text = get_document_text(uploaded_file)
                        if raw_text:
                            text_chunks = get_text_chunks(raw_text)
                            get_vector_store(text_chunks, uploaded_file.name)
                            st.success(f"Processed {uploaded_file.name}")
                            logging.info(f"Document {uploaded_file.name} processing completed successfully.")
                st.rerun() # Rerun to update the list of available documents
            else:
                st.warning("Please upload documents to process.")
                logging.warning("Process Uploaded Documents button clicked without files.")

        st.subheader("Available Documents")
        # Display available documents from Chroma DB
        available_docs = get_uploaded_files_from_vectordb()
        if available_docs:
            for doc_name in available_docs:
                st.write(f"- {doc_name}")
        else:
            st.info("No documents uploaded yet.")

    # Main area for user interaction
    # Input field for user's question
    # This part is now redundant as the main chat input handles both text and voice
    # if prompt := st.chat_input("Ask me anything about the uploaded documents!"):
    #     st.session_state.messages.append({"role": "user", "content": prompt})
    #     with st.chat_message("user"):
    #         st.markdown(prompt)

    #     with st.chat_message("assistant"):
    #         with st.spinner("Thinking..."):
    #             # Prepare chat history for the model
    #             # Exclude the current user prompt from chat history passed to the model
    #             chat_history_str = "\n".join([f"{msg['role']}: {msg['content']}" for msg in st.session_state.messages[:-1]])
    #             response = user_input(prompt, chat_history_str)
    #             st.markdown(response)
    #     st.session_state.messages.append({"role": "assistant", "content": response})

# Entry point of the application
if __name__ == "__main__":
    main()