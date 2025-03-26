from dotenv import load_dotenv
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os 



from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate

load_dotenv()

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))


# getting the pdf files  and extracting the text
def get_pdf_text(pdf_docs):
    text=""
    for pdf in pdf_docs:
        pdf_reader=PdfReader(pdf)
        for page in pdf_reader.pages:
            text+=page.extract_text()
    return text

# contverting the whole text we get from the above function into the smaller chunks
def get_text_chunk(text):
    text_splitter=RecursiveCharacterTextSplitter(chunk_size=10000,chunk_overlap=1000)
    chunks=text_splitter.split_text(text)
    return chunks


# getting the vector store to save the data in the faiss_store files where i can see the data
def get_vector_store(text_chunks):
    embeddings=GoogleGenerativeAIEmbeddings(model="models/embeddings-001")
    vector_store=FAISS.from_text(text_chunks,embeddings=embeddings)
    vector_store.save_local("faiss_store")




def get_conversational_chain():
    PromptTemplate="""
      Answer the questions as long as can you with the details explanation of that particular questions ,make sure to provide  detailed explanation of the questions,if dont understand the questions just say ,"I dont know the answer of this question",dont provide the wrong answers
      Context:\n {context}?\n
      Question:\n {question}\n

      Answer:


    """
    model=ChatGoogleGenerativeAI(model="gemini-2.0-flash",temperature=0.5)

    prompt=PromptTemplate(template=PromptTemplate,input_variables={"context","Question"})

    chain=load_qa_chain(model,chain_type="stuff",prompt=prompt)
    return chain


def user_input(user_questions):
    embeddings=GoogleGenerativeAIEmbeddings(model="models/embeddings-001")

    new_db = FAISS.load_local("faiss_index",embeddings)
    docs= new_db.simiarity_search(user_questions)

    chain=get_conversational_chain()


    response=chain(
        {"input_documnets":docs , "question":user_questions}
        ,return_only_output=True
    )


    print(response)
    st.write("Reply:",response["output"])


def main():
    st.set_page_config("Chat With Multiple PDF")
    st.header("Chat with Multi PDF using Gemini")

    user_question=st.text_input("Ask a question:")




    if user_question:
        user_input(user_question)

    with st.sidebar:
        st.title("Upload PDF Files")
        pdf_docs = st.file_uploader("Upload your PDF Files and Click on the button below", accept_multiple_files=True)

        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunk(raw_text)
                get_vector_store(text_chunks)
                st.success("PDF text extracted successfully!")

if __name__ == "__main__":
    main()