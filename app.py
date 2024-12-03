import streamlit as st
import os
import nltk
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import NLTKTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.messages import SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts import HumanMessagePromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# Read API Key
key = os.environ["GOOGLE_API_KEY"]

# Prompt Template
system_template = """
                You are a Helpful AI Bot.
                Given a context and question from user,
                you should answer based on the given context.
                """

user_template = """
        Answer the question based on the given context.
        Context: {context}
        Question: {question}
        Answer:
        """

# Load chat model
chat_model = ChatGoogleGenerativeAI(google_api_key=key,
                                    model="gemini-1.5-flash")

# Load the doc
loader = PyPDFLoader("./assets/empirical_comparison.pdf")
pages = loader.load_and_split()

# Split document into chunks
nltk.download('punkt_tab')
text_splitter = NLTKTextSplitter(chunk_size=500, chunk_overlap=100)
chunks = text_splitter.split_documents(pages)

# Create chunk embedding
embedding_model = GoogleGenerativeAIEmbeddings(
    google_api_key=key,
    model="models/text-embedding-004")

# Embed each chunk and load it into the vector store and persist it
db = Chroma.from_documents(chunks, embedding_model,
                           persist_directory="./chroma_db_")

# Set up connection with ChromaDB
db_connection = Chroma(
    persist_directory="./chroma_db_", embedding_function=embedding_model
)

# Convert ChromaDB db_connection to Retriever Object
retriever = db_connection.as_retriever(search_kwargs={"k": 5})

# Setup chat template
chat_template = ChatPromptTemplate.from_messages(
    [
        # System Message Prompt Template
        SystemMessage(
            content=system_template
        ),
        # Human Message Prompt Template
        HumanMessagePromptTemplate.from_template(
            user_template
        ),
    ]
)

# Setup output parser
output_parser = StrOutputParser()


# Setup rag chain
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | chat_template
    | chat_model
    | output_parser
)

# Streamlit UI
with st.sidebar:
    st.title("""
             :blue[RAG on the'Empirical Comparison' Paper]
             """)
st.title(":blue[💬Document Chatbot]")
query = st.text_area("Enter your query:",
                     placeholder="Enter your query here...",
                     height=100)

if st.button("Submit Your Query"):
    if query:
        response = rag_chain.invoke(query)
        st.write(response)
    else:
        st.warning("Please enter a question.")
