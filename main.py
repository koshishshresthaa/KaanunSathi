import streamlit as st
import chromadb
from generator import Generation
from langchain_chroma import Chroma
from models import llm,embedding_model
from streamlit_chat import message
from collectionrouter import get_contextname_collection
import time

# chromavectorstore=Chroma(persist_directory="/kaggle/input/laboractvectorstore/vectorstore", embedding_function=embedding_model)

#creating retirever
# retriever=chromavectorstore.as_retriever()

#pulling persistent chroma client again and only refering to the collection according to the query
# chroma_persistent_client = chromadb.PersistentClient(path="/kaggle/working/nepali-law-acts/finalvectorstore")

        
FAQs=["Select a Question",
    "1. I work overtime at my office, what are the benefits I will get after working overtime and what is the standard nomral timings?",
      "2. Give me list of Provisions Relating to Leave",
      "3. According to Election Commission of Nepal , What are the Election Code of Conduct ? ",
     "4. What are the Functions, Duties and Powers of Election Commission in Nepal ?",
      "5. Say me about different types of passport in Nepal ?",
      "6. What might be the reasons of my passport being cancelled?"
     ]



def main():
    '''
    Main function to run the chatbot.
    '''
    
    
    st.title("KaanunSathi")
    
    st.sidebar.title("FAQs")
    if "selected_question" not in st.session_state:
        st.session_state.selected_question = "Select a Question"

    selected_question = st.sidebar.selectbox("Select a Question", FAQs, index=FAQs.index(st.session_state.selected_question))
                    
                
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Accept user input
    
    prompt = None

    
    if selected_question != "Select a Question":
        prompt= selected_question
        st.session_state.selected_question = "Select a Question"
        
    else:
        
        prompt= st.chat_input("Your message here")
        
      
    if prompt:
        if prompt.lower()=="clear":
            st.session_state.messages.clear()
            with st.chat_message("assistant"):
                st.write("Chat history cleared.")
    
        else:
            context_collection_name=get_contextname_collection(prompt,st.session_state.messages)
            
            if context_collection_name=="Invalid":
                
                with st.chat_message("user"):
                    st.markdown(prompt)
                    
                with st.chat_message("assistant"):
                    st.write("Your query appears to be invalid or out of context, and I am unable to provide an answer based on the information provided. Please refine your query and try again.")
  
            else:

                #refering to only that collection of the client according to user query
                chromavectorstore= Chroma(
                    persist_directory="./lawfinalvectorstore",
                    collection_name=context_collection_name,
                    embedding_function=embedding_model,
                )


                retriever=chromavectorstore.as_retriever()

                generate=Generation(llm)
     
                # Add user message to chat history
                st.session_state.messages.append({"role": "user", "content": prompt})
                # Display user message in chat message container
                with st.chat_message("user"):
                    st.markdown(prompt)
                    
                with st.expander("See Relevant Collection"):
                    st.write("The most relevant collection for the query is : ",context_collection_name)
                
                
                # Display the retrieved context in an expandable section
                with st.expander("See Retrieved Context"):
                   context = retriever.get_relevant_documents(prompt)
                   st.write(context)



                # Display assistant response in chat message container
                with st.chat_message("assistant"):
                    response = st.write_stream(generate.generate_answer(st.session_state.messages, prompt,retriever))
                # Add assistant response to chat history
                st.session_state.messages.append({"role": "assistant", "content": response})
                
     # Display the note in gray and center-aligned
    note_html = """
    <div style="text-align: center; color: gray; margin-top: 20px;">
        Note: KaanunSathi can make mistakes. Verify yourself for relevant info.
    </div>
    """
    st.markdown(note_html, unsafe_allow_html=True)

    
if __name__ == "__main__":
    main()
