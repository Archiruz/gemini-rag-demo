import streamlit as st
import os
import nltk
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import NLTKTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
# from langchain_core.messages import SystemMessage
from langchain_core.prompts import ChatPromptTemplate
# from langchain_core.prompts import HumanMessagePromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough


class RAGChat:
    def __init__(self, pdf_path: str):
        self.pdf_path = pdf_path
        self.api_key = os.environ["GOOGLE_API_KEY"]
        self.chat_model = None
        self.embedding_model = None

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
            Given a context and question from user,
            you should answer based on the given context.
            """
        self.user_template = """
            Answer the question based on the given context.
            Context: {context}
            Question: {question}
            Answer:
            """

    def process_document(self):
        """Load and process the PDF document"""
        # Check if the embedding model is initialized
        if self.embedding_model is None:
            self.intitialize_embedding()

        # Load document
        loader = PyPDFLoader(self.pdf_path)
        pages = loader.load_and_split()

        # Split into chunks
        nltk.download("punkt")
        text_splitter = NLTKTextSplitter(chunk_size=500, chunk_overlap=100)
        chunks = text_splitter.split_documents(pages)

        # Embed each chunk and load it into the vector store and persist it
        Chroma.from_documents(chunks, self.embedding_model,
                              persist_directory="./chroma_db_")
        print("Document processed.")

        # Check if the database is created
        if not os.path.exists("./chroma_db_"):
            raise ValueError("ChromaDB folder not found.")

        print("Database created.")

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
        db_connection = Chroma(
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
        retriever = db_connection.as_retriever(search_kwargs={"k": 5})

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
    rag_chat = RAGChat("./assets/empirical_comparison.pdf")

    st.title("RAG Gemini Demo")

    if "session_initialized" not in st.session_state:
        with st.spinner("Initializing system..."):
            rag_chat.process_document()
            st.session_state.session_initialized = True

    if st.session_state.session_initialized:
        query = st.text_area("Enter your query:",
                             placeholder="Enter your query here...",
                             height=100)

        if st.button("Submit Your Query"):
            if query:
                rag_chat.initialize_templates()
                response = rag_chat.run_query(query)
                st.write(response)
            else:
                st.warning("Please enter a question.")


if __name__ == "__main__":
    main()
