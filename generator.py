import time
from langchain_core.prompts import PromptTemplate
from langchain.chains import create_history_aware_retriever,  create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_core.chat_history import BaseChatMessageHistory
from langchain import hub
import streamlit as st
class Generation:
    '''
    
    Generation of the output based on retrieved context and user query.
    '''
    
    def __init__(self,llm):
        self.llm=llm
        
    # Post-processing
    def format_docs(self,docs):
        return "\n\n".join(doc.page_content for doc in docs)
    
    def create_chain(self,retriever):
        
        '''
        Function to create the retrieval chain with history
        '''
        
        #normal prompt 
        prompt = ChatPromptTemplate.from_messages([
            ("system", """
         As an expert in the laws of Nepal, please provide a detailed and well-reasoned answer to the following input question. 
         While first interaction, greet user with "Hello! I'm your KaanunSathi. How can I assist you today with your queries related to Nepali laws and acts? Please feel free to ask any questions you have." only during the first interaction. Do not greet the user in subsequent replies. Always reply the user in a polite and humble manner. 
         Use the chain of thought reasoning to break down the query and explain each step and your thoughts leading to the conclusion.
         Don't display the thought process, just think on your own.
         Don't mention section if you are not 100% sure about it or else it will lead to wrong section number which becomes blunder mistake.
         Also ensure your final output is always displayed separately.
         Ensure your answer is strictly based on the provided context only and inside of the given context.
         If the question is of different laws that is not related to the provided context and also of entirely different topic then simply say "Sorry, I don't know the answer to that question!"
         You are strictly prohibited to reply on your own if the question is out of context , donot generate answers at all.

        {context}

        Question: {input}

        Helpful Answer:"""),
            MessagesPlaceholder(variable_name="chat_history"),
            ("user", "{input}")
        ])
        
        #rag chain without history
        chain = create_stuff_documents_chain(
            llm=self.llm,
            prompt=prompt
        )
        
        #prompt after adding chat_history
        contextualize_q_system_prompt = """Given a chat history and the latest user question \
        which might reference context in the chat history, formulate a standalone question \
        which can be understood without the chat history. Do NOT answer the question, \
        just reformulate it if needed and otherwise return it as is."""
        
        prompt_with_history = ChatPromptTemplate.from_messages(
             [
                 ("system", contextualize_q_system_prompt),
                 MessagesPlaceholder("chat_history"),
                 ("human", "{input}"),
             ]
        )
        
        #creating history aware retriever 
        history_aware_retriever = create_history_aware_retriever(
            llm=self.llm,
            retriever=retriever,
            prompt=prompt_with_history
        )

        
        #creating history aware retriever chain
        retrieval_chain = create_retrieval_chain(
            history_aware_retriever,
            chain
        )

        return retrieval_chain
    

    def generate_answer(self,chat_history,query,retriever):
        '''
        Function to generate the output.
        '''
        
       
        chain=self.create_chain(retriever)
    
        
        response = chain.invoke({
        "chat_history": chat_history,
        "input": query,
        })
#         return response["answer"]
    
        for word in response['answer'].split():
            yield word + " "
            time.sleep(0.05)
    
    
