from flask import Flask, render_template, request, session
import openai
import os
from langchain.chains import RetrievalQA
from langchain.memory import ConversationBufferMemory
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

app = Flask(__name__)
app.secret_key = 'supersecretkey'

######################################
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain.prompts import ChatPromptTemplate
from langchain.llms import Ollama
from langchain.embeddings import OllamaEmbeddings
from langchain.vectorstores import FAISS
from langchain.docstore.document import Document
from langchain.text_splitter import CharacterTextSplitter
import csv

# Global variables to store initialized components
llm = None
retrieval_chain = None
context = []

def init():
    global llm, retrieval_chain, context
    # Initialize the Ollama model
    llm = Ollama(model='llama3.1:70b')
    
    # Build a list of documents, each containing a piece of text content
    docs = []
    with open('nursing_113.csv', newline='', encoding='utf-8') as csvfile:
        csv_reader = csv.reader(csvfile)
        for row in csv_reader:
            # Assuming each row contains a single text entry
            content = row[0]  # Modify index if there are multiple columns
            docs.append(Document(page_content=content))
    
    # Set up the text splitter
    text_splitter = CharacterTextSplitter(chunk_size=20, chunk_overlap=5)
    documents = text_splitter.split_documents(docs)
    
    # Initialize embeddings
    embeddings = OllamaEmbeddings(model="llama3.1:70b")
    
    # Build the vector database using FAISS
    vectordb = FAISS.from_documents(documents, embeddings)
    
    # Set the vector database as the retriever
    retriever = vectordb.as_retriever()
    
    # Set up the prompt template
    prompt = ChatPromptTemplate.from_messages([
        ('system', '请根据以下内容回答用户的问题：\n\n{context}'),
        ('user', '问题：{input}'),
    ])
    
    # Create the document chain
    document_chain = create_stuff_documents_chain(llm, prompt)
    
    # Create the retrieval chain
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    
    # Initialize context
    context = []

# def get_answer(input_text):
#     global retrieval_chain, context
#     response = retrieval_chain.invoke({
#         'input': input_text,
#         'context': context
#     })
#     context = response.get('context', [])
#     return response['answer']

# Example usage:
# Initialize the system (do this once)
init()
#############################################

##########
from langchain.llms import Ollama  # 使用 LangChain 的 Ollama 模型
from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
prompt = ChatPromptTemplate(
    messages=[
        SystemMessagePromptTemplate.from_template(
            "You are a nice chatbot having a conversation with a human."
        ),
        MessagesPlaceholder(variable_name="chat_history"),
        HumanMessagePromptTemplate.from_template("{question}")
    ]
)
# 設定回傳對話歷史
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
conversation = LLMChain(
    llm=llm,
    prompt=prompt,
    verbose=True,
    memory=memory
)

# 定義 get_answer 函數
# def get_answer(input_text):
#     response = conversation({"question": input_text})
#     return response["text"]
def get_answer(input_text):
    global retrieval_chain, context
    response = retrieval_chain.invoke({
        'input': input_text,
        'context': context
    })
    context = response.get('context', [])
    return response['answer']
#########

messages = []

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

def ExplainMore():
    return get_answer("Please explain more in detail. Answer in English.")

def OriginalCase():
    return get_answer("Please tell me: What is the original case for the current discussion? Answer in English.")

def SimilarScenario():
    return get_answer("Please generate a similar scenario based on our current discussion for me to practice. Answer in English.")

def RelevantTheories():
    return get_answer("Please tell me: What data and theories form the basis of our current discussion, and provide the sources of this information. Answer in English.")

@app.route('/main', methods=['GET', 'POST'])
def main():
    if request.method == 'POST':
        input_text = request.form['input_text']
        previous_input = session.get('previous_system_response', '')
        
        session['previous_system_response'] = input_text
        
        if 'Explain More' in input_text:
            result_text = ExplainMore()
        elif 'Original Case' in input_text:
            result_text = OriginalCase()
        elif 'Similar Scenario' in input_text:
            result_text = SimilarScenario()
        elif 'Relevant Theories' in input_text:
            result_text = RelevantTheories()
        else:
            # prompt_template = f"Answer in English. Based on my input, please determine my intent. If my input doesn't mention 'generate a new case' or similar terms, it means I am providing feedback on an existing case. You should compare my feedback with previous data handling methods, then generate suggestions and data sources. If I mention 'generate a new case,' then please generate a new case description, and after finishing the description, ask: Based on this case description, what will your next decision and action be?\n\n You don't need to explain your reasoning, just start outputting the relevant content. If it's Intent 1, this is the last system message (case description): {previous_input}. If it's Intent 2, please ignore the last system message and generate a new case description.\n\n This is my input:"
            prompt_template = ""
            result_text = get_answer(prompt_template + input_text)
        
        session['previous_system_response'] = result_text
        
        messages.append({'text': input_text, 'type': 'user-message'})
        messages.append({'text': result_text, 'type': 'bot-message'})
        return render_template('main.html', messages=messages)

    return render_template('main.html', messages=messages)

@app.route('/data')
def data():
    return render_template('data.html')

if __name__ == '__main__':
    app.run(debug=True)
