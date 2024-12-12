__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import streamlit as st
import os
import nltk
import shutil
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import NLTKTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
# from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from chromadb.api.client import SharedSystemClient


class RAGChat:
    def __init__(self):
        self.api_key = os.environ["GOOGLE_API_KEY"]
        self.chat_model = None
        self.embedding_model = None
        self.vector_db = None

    def initialize_model(self):
        """Initialize the Gemini model"""
        self.chat_model = ChatGoogleGenerativeAI(
            google_api_key=self.api_key, model="gemini-1.5-flash"
        )
        # if self.chat_model:
        #     print("Model initialized.")

    def intitialize_embedding(self):
        """Initialize the embedding model"""
        self.embedding_model = GoogleGenerativeAIEmbeddings(
            google_api_key=self.api_key,
            model="models/text-embedding-004")
        # if self.embedding_model:
        #     print("Embedding model initialized.")

    def initialize_templates(self):
        """Initialize the system and user templates"""
        self.system_template = """
            You are a Helpful AI Bot created by Alvian.
            You can reply to a small talk.
            You will reply with a Markdown formatted response.
            Given a context and question from user,
            you should answer based on the given context.
            """
        self.user_template = """
            Answer the question based on the given context.
            Context: {context}
            Question: {question}
            Answer:
            """

    def process_document(self, pdf):
        """Load and process the PDF document"""
        if pdf is not None:

            # Load document
            loader = PyPDFLoader(pdf)
            pages = loader.load_and_split()

            # Split into chunks
            nltk.download("punkt_tab")
            text_splitter = NLTKTextSplitter(chunk_size=1000,
                                             chunk_overlap=200)
            chunks = text_splitter.split_documents(pages)

            # Check if the embedding model is initialized
            if self.embedding_model is None:
                self.intitialize_embedding()

            # Embed each chunk and load it into the vector store and persist it
            self.vector_db = Chroma.from_documents(
                chunks,
                self.embedding_model,
                persist_directory="./chroma_db_")
            print("Document processed.")

    def delete_chroma_db(self):
        """Delete the ChromaDB"""
        if os.path.exists("./chroma_db_"):
            shutil.rmtree("./chroma_db_")
            print("ChromaDB folder deleted.")

    def format_docs(self, docs):
        return "\n\n".join(doc.page_content for doc in docs)

    def run_query(self, query: str):
        """Run query on the database"""
        # Check vector database
        if not os.path.exists("./chroma_db_"):
            raise ValueError("ChromaDB folder not found.")

        # Check if the model is initialized
        if self.chat_model is None:
            self.initialize_model()

        # Check if the embedding model is initialized
        if self.embedding_model is None:
            self.intitialize_embedding()

        # Set up connection with ChromaDB
        self.vector_db = Chroma(
            persist_directory="./chroma_db_",
            embedding_function=self.embedding_model)
        # print("Connection established.")

        # Prompt template
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", self.system_template),
                ("human", self.user_template)
            ]
        )
        retriever = self.vector_db.as_retriever(search_kwargs={"k": 5})

        if not retriever:
            raise ValueError("Retriever setup failed.")

        retrieval_chain = (
            {
                "context": retriever | self.format_docs,
                "question": RunnablePassthrough(),
            }
            | prompt
            | self.chat_model
            | StrOutputParser()
        )

        try:
            response = retrieval_chain.invoke(query)
            return response
        except Exception as e:
            return f"An error occurred: {e}"


def main():
    rag_chat = RAGChat()

    st.title("RAG on Gemini Demo by [Alvian](https://archiruz.github.io)")
    st.write("""
             This is a demo of using RAG on Gemini
             to answer questions based on a PDF document.
             """)
    st.subheader("Upload a PDF and ask a question to Gemini.")

    uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])
    upload_button = st.button("Upload PDF")

    if 'pdf_uploaded' not in st.session_state:
        st.session_state.pdf_uploaded = False

    if 'pdf' not in st.session_state:
        st.session_state.pdf = None

    if upload_button:
        with st.spinner("Processing pdf..."):
            rag_chat.delete_chroma_db()
            # Save the file temporarily
            temp_file = "./temp.pdf"
            with open(temp_file, "wb") as f:
                f.write(uploaded_file.getvalue())
                st.session_state.pdf = uploaded_file.name

            rag_chat.process_document(temp_file)
            # Close connection after processing
            rag_chat.vector_db._client._system.stop()
            SharedSystemClient.clear_system_cache()
            rag_chat.vector_db = None
            st.session_state.pdf_uploaded = True
            st.rerun()

    if st.session_state.pdf_uploaded:
        with st.form(key="query_form"):
            query = st.text_area(f"Enter your query to {st.session_state.pdf}",
                                 placeholder="Enter your query here...",
                                 height=100)
            submit_button = st.form_submit_button(label='Submit')

            if submit_button:
                if query:
                    with st.spinner("Processing..."):
                        rag_chat.initialize_templates()
                        response = rag_chat.run_query(query)
                        # Close connection after processing
                        rag_chat.vector_db._client._system.stop()
                        SharedSystemClient.clear_system_cache()
                        rag_chat.vector_db = None
                    st.write(response)
                else:
                    st.warning("Please enter a query.")


if __name__ == "__main__":
    main()
