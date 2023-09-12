import os
import base64
import fitz
from dotenv import load_dotenv
import streamlit as st
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import AzureChatOpenAI
import openai
from langchain.chat_models import AzureChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import AzureSearch
from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter


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
def get_vectorstore(directory_path, chunk_size=1000, chunk_overlap=20, separator="\n"):
    """Load PDF document and create a vectorstore from the text chunks.
    Cache the result to avoid repeated calls to the embeddings model."""

    # Configure OpenAI API
    openai.api_type = "azure"
    openai.api_base = os.getenv('AZURE_OPENAI_ENDPOINT')
    openai.api_key = os.getenv('AZURE_OPENAI_KEY')
    openai.api_version = os.getenv('AZURE_OPENAI_API_VERSION')

    embeddings = OpenAIEmbeddings(
        deployment=os.getenv("OPENAI_EMBEDDINGS_DEPLOYMENT_NAME"),
        chunk_size=1,
        request_timeout=120,
    )

    st.write(os.getenv('AZURE_SEARCH_SERVICE'))

    # Connect to Azure Cognitive Search
    acs = AzureSearch(azure_search_endpoint=f"https://{os.getenv('AZURE_SEARCH_SERVICE')}.search.windows.net",
                    azure_search_key=os.getenv('AZURE_SEARCH_KEY'),
                    index_name=os.getenv('AZURE_SEARCH_INDEX'),
                    embedding_function=embeddings.embed_query)
    docs = load_docs(directory_path)
    print(f"Number of documents: {len(docs)}")
    # text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
    #     chunk_size=chunk_size, chunk_overlap=chunk_overlap, separator=separator
    # )
    # texts = text_splitter.split_documents(docs) 
    texts = split_docs(docs, chunk_size, chunk_overlap)
    print(f"Number of chunks: {len(texts)}")
    # Add documents to Azure Search
    acs.add_documents(documents=docs)


def load_docs(directory):
    loader = DirectoryLoader(directory)
    documents = loader.load()
    return documents


def main():
    load_dotenv()

    # set page config and slide bar
    st.set_page_config(layout="wide", page_title="Add your data...", page_icon="🔍")
    # set_sidebar()

    st.write("<h1><center>Add your customize data</center></h1>", unsafe_allow_html=True)

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

                st.write("Start to generate index to Azure Search...")
                get_vectorstore(
                folder_path
                )
                st.write("Your index is ready! Please go to the chat page to start your conversation.")


if __name__ == "__main__":
    main()
