import PyPDF2
from langchain_community.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ChatMessageHistory, ConversationBufferMemory
import chainlit as cl
from langchain_groq import ChatGroq
from dotenv import load_dotenv


load_dotenv()


my_groq_api_key = "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"


groq_llm = ChatGroq(
    groq_api_key=my_groq_api_key, model_name="mixtral-8x7b-32768",
    temperature=0.2
)

@cl.on_chat_start
async def start_chat():
    uploaded_files = None  

    
    while uploaded_files is None:
        uploaded_files = await cl.AskFileMessage(
            content="Please upload a PDF file to start!",
            accept=["application/pdf"],
            max_size_mb=100,  
            timeout=180,  
        ).send()

    pdf_file = uploaded_files[0]  
    print(pdf_file)  

    
    processing_msg = cl.Message(content=f"Processing `{pdf_file.name}`...")
    await processing_msg.send()

    
    pdf_reader = PyPDF2.PdfReader(pdf_file.path)
    pdf_text_content = ""
    for page in pdf_reader.pages:
        pdf_text_content += page.extract_text()

    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=50)
    text_chunks = text_splitter.split_text(pdf_text_content)

    
    chunk_metadatas = [{"source": f"{i}-chunk"} for i in range(len(text_chunks))]

    
    embedding_model = OllamaEmbeddings(model="nomic-embed-text")
    vector_store = await cl.make_async(Chroma.from_texts)(
        text_chunks, embedding_model, metadatas=chunk_metadatas
    )

    
    message_history = ChatMessageHistory()

    
    conversation_memory = ConversationBufferMemory(
        memory_key="chat_history",
        output_key="answer",
        chat_memory=message_history,
        return_messages=True,
    )

    
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=groq_llm,
        chain_type="stuff",
        retriever=vector_store.as_retriever(),
        memory=conversation_memory,
        return_source_documents=True,
    )

    
    processing_msg.content = f"Processing `{pdf_file.name}` completed. You can now ask questions!"
    await processing_msg.update()

    
    cl.user_session.set("conversation_chain", conversation_chain)

@cl.on_message
async def handle_message(message: cl.Message):
    
    conversation_chain = cl.user_session.get("conversation_chain")


    callback_handler = cl.AsyncLangchainCallbackHandler()

    
    response = await conversation_chain.ainvoke(message.content, callbacks=[callback_handler])
    answer_text = response["answer"]
    source_docs = response["source_documents"]

    text_elements = []  

    
    if source_docs:
        for idx, doc in enumerate(source_docs):
            doc_name = f"source_{idx}"
            
            text_elements.append(
                cl.Text(content=doc.page_content, name=doc_name)
            )
        doc_names = [text_el.name for text_el in text_elements]

        
        if doc_names:
            answer_text += f"\nSources: {', '.join(doc_names)}"
        else:
            answer_text += "\nNo sources found"
    
    
    await cl.Message(content=answer_text, elements=text_elements).send()
