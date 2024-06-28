import streamlit as st
import pickle
import os
from langchain_google_genai import GoogleGenerativeAI
from langchain.chains.qa_with_sources.retrieval import RetrievalQAWithSourcesChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredURLLoader
from langchain.vectorstores import FAISS
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import faiss
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain.docstore.document import Document

load_dotenv()  # Take environment variables from .env (especially google API key)

st.title("News Research Tool")
st.sidebar.title("News Article URLs")

urls = []
for i in range(3):
    url = st.sidebar.text_input(f"URL {i+1}")
    urls.append(url)

process_url_clicked = st.sidebar.button("Process URLs")
index_file_path = "faiss_index.bin"
docstore_file_path = "docstore.pkl"

main_placeholder = st.empty()
llm = GoogleGenerativeAI(api_key=os.getenv("GOOGLE_API_KEY"), model="gemini-1.5-pro")

if process_url_clicked:
    loader = UnstructuredURLLoader(urls=urls)
    main_placeholder.text("Data Loading...Started...")
    data = loader.load()

    # Split data
    text_splitter = RecursiveCharacterTextSplitter(separators=['\n\n', '\n', '.', ','], chunk_size=1000)
    main_placeholder.text("Text Splitting...Started...")
    docs = text_splitter.split_documents(data)

    for doc in docs:
        if not isinstance(doc.page_content, str):
            doc.page_content = str(doc.page_content)

    st.session_state["docs"] = docs

    # Embed documents
    model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
    embeddings = model.encode([doc.page_content for doc in docs])

    # Create FAISS index
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

    # Create InMemoryDocstore
    docstore = InMemoryDocstore({str(i): doc for i, doc in enumerate(docs)})

    # Store the FAISS index and document store
    faiss.write_index(index, index_file_path)
    with open(docstore_file_path, "wb") as f:
        pickle.dump(docstore._dict, f)
    st.sidebar.success("URLs processed and data stored successfully.")
    main_placeholder.text("Embedding Vectors Built and Indexed.")

query = main_placeholder.text_input("Question: ")
if query:
    if os.path.exists(index_file_path) and os.path.exists(docstore_file_path):
        try:
            # Load FAISS index and document store
            index = faiss.read_index(index_file_path)
            with open(docstore_file_path, "rb") as f:
                docstore_dict = pickle.load(f)
            docstore = InMemoryDocstore(docstore_dict)
            st.session_state["docs"] = list(docstore_dict.values())

            # Recreate the SentenceTransformer model
            model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

            # Create FAISS retriever
            vectorstore = FAISS(embedding_function=model.encode, index=index, docstore=docstore,
                                index_to_docstore_id={i: str(i) for i in range(len(st.session_state["docs"]))})

            chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=vectorstore.as_retriever())
            result = chain({"question": query}, return_only_outputs=True)
            
            st.header("Answer")
            st.write(result["answer"])

            # Display sources, if available
            sources = result.get("sources", "")
            if sources:
                st.subheader("Sources:")
                sources_list = sources.split("\n")  # Split the sources by newline
                for source in sources_list:
                    st.write(source)
        except Exception as e:
            st.error(f"Error retrieving answer: {e}")
    else:
        st.error("FAISS store not found. Please process URLs first.")
