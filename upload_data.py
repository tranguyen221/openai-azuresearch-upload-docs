import os
import base64
import fitz
from dotenv import load_dotenv
load_dotenv()
import streamlit as st
import openai
from langchain.embeddings import OpenAIEmbeddings
from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import AzureSearch
from form_recognizer_loader import FormRecognizerLoader


def display_pdf(placeholder, pdf_bytes, page_number=None):
    """Display PDF in Streamlit app."""
    pn_str = "" if page_number is None else f"#page={page_number}"
    base64_pdf = base64.b64encode(pdf_bytes).decode("utf-8")
    # Embedding PDF in HTML
    pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}{pn_str}" width="100%" height="800" type="application/pdf"></iframe>'
    # Displaying File
    placeholder.markdown(pdf_display, unsafe_allow_html=True)

def split_docs(documents, chunk_size=1000, chunk_overlap=20):
  text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
  docs = text_splitter.split_documents(documents)
  return docs

@st.cache_data
def load_docs(directory_path):
    loader = DirectoryLoader(directory_path)
    documents = loader.load()
    return documents

@st.cache_resource
def index_documents(directory_path, chunk_size=1000, chunk_overlap=20, separator="\n"):
    """Load PDF document and create a vectorstore from the text chunks.
    Cache the result to avoid repeated calls to the embeddings model."""
    st.write("Chunking documents...")
    docs = load_docs(directory_path)
    print(f"Number of documents: {len(docs)}")

    openai.api_type = "azure"
    openai.api_base = os.getenv("AZURE_OPENAI_BASE")
    # openai.api_key = os.getenv("AZURE_OPENAI_EMBEDDING_KEY")
    # openai.api_version = os.getenv("AZURE_OPENAI_API_VERSION")

    openai.api_key = os.getenv("AZURE_OPENAI_EMBEDDING_KEY_BACKUP")
    openai.api_version = os.getenv("AZURE_OPENAI_API_VERSION")

    embeddings = OpenAIEmbeddings(
            deployment_id=os.getenv("AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT_NAME_BACKUP"),
            chunk_size=1,
            request_timeout=120,
        )
    # Connect to Azure Cognitive Search
    acs = AzureSearch(azure_search_endpoint=f"https://{os.getenv('AZURE_SEARCH_SERVICE')}.search.windows.net",
                azure_search_key=os.getenv('AZURE_SEARCH_KEY'),
                index_name=os.getenv('AZURE_SEARCH_INDEX'),
                embedding_function=embeddings.embed_query)
 
    texts = split_docs(docs, chunk_size, chunk_overlap)
    print(f"Number of chunks: {len(texts)}")
    st.write("Starting to generate index in Azure Search...")
    # Add documents to Azure Search
    acs.add_documents(documents=docs)
    st.write("Your index is ready! Please go to the chat page to start your conversation.")
    st.markdown("[Go to chat page](https://hackathon-openai-ip-reuse.azurewebsites.net/)")


def load_docs(directory):
    loader = DirectoryLoader(directory)
    documents = loader.load()
    return documents


def main():
    # set page config and slide bar
    st.set_page_config(layout="wide", page_title="Add your data...", page_icon="🔍")
    # set_sidebar()

    st.write("<h1><center>Add your customized data</center></h1>", unsafe_allow_html=True)

    st.subheader("Upload document")

    # Upload a PDF file
    uploaded_file = st.file_uploader(
        "Select your data to upload", type=["pdf"]
    )
    if uploaded_file is not None:
        pdf_bytes = uploaded_file.read()
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        folder_path = "data/tmp"
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        doc.save(f"{folder_path}/{uploaded_file.name}")

        left_column, right_column = st.columns(2)
        with left_column:
            st.subheader("Uploaded PDF")
            pdf_placeholder = st.empty()
            display_pdf(pdf_placeholder, pdf_bytes)
            

        with right_column:
            st.subheader("Add data")
            st.write("""
            Your data is used to help ground the model with specific data. Please click on the Run button to generate index to Azure Search. 
                     Your data is stored securely in your Azure subscription.
            """)
            run_button = st.button("Run")
            
            if run_button:
                
                index_documents(
                    folder_path
                )


if __name__ == "__main__":
    main()
