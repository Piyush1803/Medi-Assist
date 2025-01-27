from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.prompts import PromptTemplate
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import CTransformers
from langchain.chains import RetrievalQA
import chainlit as cl
import logging
import speech_recognition as sr

DB_FAISS_PATH = 'db'

# Set up logging
logging.basicConfig(level=logging.DEBUG)

custom_prompt_template = """Use the following pieces of information to answer the user's question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context: {context}
Question: {question}

Only return the helpful answer below and nothing else.
Helpful answer:
"""

def set_custom_prompt():
    """
    Prompt template for QA retrieval for each vectorstore
    """
    prompt = PromptTemplate(template=custom_prompt_template,
                            input_variables=['context', 'question'])
    return prompt

# Retrieval QA Chain
def retrieval_qa_chain(llm, prompt, db):
    qa_chain = RetrievalQA.from_chain_type(llm=llm,
                                           chain_type='stuff',
                                           retriever=db.as_retriever(search_kwargs={'k': 2}),
                                           return_source_documents=True,
                                           chain_type_kwargs={'prompt': prompt}
                                           )
    return qa_chain

# Loading the model
def load_llm():
    # Load the locally downloaded model here
    llm = CTransformers(
        model="TheBloke/Llama-2-7B-Chat-GGML",
        model_type="llama",
        max_new_tokens=512,
        temperature=0.5
    )
    return llm

# QA Model Function
def qa_bot():
    print("Initializing embeddings...")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",
                                       model_kwargs={'device': 'cpu'})
    db = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
    print("Loaded FAISS database.")
    llm = load_llm()
    print("LLM loaded.")
    qa_prompt = set_custom_prompt()
    qa = retrieval_qa_chain(llm, qa_prompt, db)
    print("QA chain set up.")
    return qa

# Output function
def final_result(query):
    qa_result = qa_bot()
    response = qa_result({'query': query})
    return response

# Chainlit code
@cl.on_chat_start
async def start():
    # Initialize the bot's chain when the chat starts
    chain = qa_bot()
    msg = cl.Message(content="Starting the bot...")
    await msg.send()
    msg = cl.Message(content="Hi, Welcome to MediAssist. What is your query?")
    await msg.send()
    cl.user_session.set("chain", chain)

@cl.on_message
async def main(message: cl.Message):
    if message.content.lower() == "voice":
        # Initiate speech recognition
        recognizer = sr.Recognizer()
        await cl.Message(content="Please speak your query.").send()
        
        with sr.Microphone() as source:
            audio_data = recognizer.listen(source)
        
        try:
            # Recognize voice input
            query = recognizer.recognize_google(audio_data)
            print(f"Voice input recognized: {query}")
            await cl.Message(content=f"Recognized query: {query}").send()
            
            # Retrieve and process the recognized query
            chain = cl.user_session.get("chain")
            if not chain:
                chain = qa_bot()
                cl.user_session.set("chain", chain)
            
            response = await chain.ainvoke(query)
            answer = response["result"]
            await cl.Message(content=answer).send()

        except sr.UnknownValueError:
            await cl.Message(content="Sorry, I couldn't understand the audio.").send()
        except sr.RequestError:
            await cl.Message(content="Sorry, there was an issue with the speech recognition service.").send()

    else:
        # Handle text input as usual
        chain = cl.user_session.get("chain")
        if not chain:
            chain = qa_bot()
            cl.user_session.set("chain", chain)
        
        response = await chain.ainvoke(message.content)
        answer = response["result"]
        await cl.Message(content=answer).send()