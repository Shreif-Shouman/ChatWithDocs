import os
import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from docx import Document
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain, RetrievalQAWithSourcesChain
from htmlTemplates import css, bot_template, user_template
from langchain_community.llms import HuggingFaceEndpoint

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_docx_text(docx_docs):
    text = ""
    for docx in docx_docs:
        doc = Document(docx)
        for para in doc.paragraphs:
            text += para.text + '\n'
    return text

def get_txt_text(txt_docs):
    text = ""
    for txt in txt_docs:
        text += txt.getvalue().decode('utf-8')
    return text

def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_vectorstore(text_chunks):
    embeddings = HuggingFaceInstructEmbeddings(model_name="sentence-transformers/all-MiniLM-l6-v2")
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

def get_conversation_chain(vectorstore):
    #api_key = os.environ['HUGGINGFACEHUB_API_TOKEN'] = st.secrets['HUGGINGFACEHUB_API_TOKEN']
    #llm = HuggingFaceEndpoint(repo_id="mistralai/Mistral-7B-Instruct-v0.2")#, token=api_key)
    llm = HuggingFaceEndpoint(repo_id="describeai/gemini")
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain

def handle_userinput(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']
    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)

def main():
    load_dotenv()
    st.set_page_config(page_title="Chat with your documents", page_icon=":books:")
    st.write(css, unsafe_allow_html=True)

    custom_html = """
        <div style="display: flex; justify-content: center; align-items: center; overflow: hidden; height: 300px; background-color: #f0f0f0;">
        <img src="https://miro.medium.com/v2/resize:fit:828/format:webp/1*U8ku0ur76OSFyAOrolT99g.png" alt="Banner Image" style="height: 300px; width: auto; min-width: 100%;">
        </div>
    """
    st.markdown(custom_html, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header("Chat with your documents :books:")
    user_question = st.chat_input("Ask a question about your documents:")
    if user_question:
        handle_userinput(user_question)

    with st.sidebar:
        st.subheader("Your documents")
        uploaded_files = st.file_uploader("Upload your documents here and click on 'Process'", accept_multiple_files=True, type=['pdf', 'docx', 'txt'])
        if st.button("Process"):
            with st.spinner("Processing"):
                raw_text = ""
                pdf_docs = [file for file in uploaded_files if file.name.endswith('.pdf')]
                docx_docs = [file for file in uploaded_files if file.name.endswith('.docx')]
                txt_docs = [file for file in uploaded_files if file.name.endswith('.txt')]

                if pdf_docs:
                    raw_text += get_pdf_text(pdf_docs)
                if docx_docs:
                    raw_text += get_docx_text(docx_docs)
                if txt_docs:
                    raw_text += get_txt_text(txt_docs)

                text_chunks = get_text_chunks(raw_text)
                vectorstore = get_vectorstore(text_chunks)
                st.session_state.conversation = get_conversation_chain(vectorstore)

if __name__ == '__main__':
    main()
