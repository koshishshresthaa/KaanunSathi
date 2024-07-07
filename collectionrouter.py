from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
from typing import Literal
from langchain_core.runnables import RunnablePassthrough,RunnableMap
from langchain_core.output_parsers import JsonOutputParser,StrOutputParser
import streamlit as st
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain import PromptTemplate
from langchain.chains import create_history_aware_retriever,  create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.utils.math import cosine_similarity
from models import embedding_model,llm



# Data model
# class RouteQuery(BaseModel):
#     """Route a user query to the most relevant datasource."""

#     datasource: Literal["TheLaborAct","PassportAct","ElectionCommissionAct"] = Field(
#         ...,
#         description="Given a user question choose which datasource would be most relevant for answering their question",
#     )

        
def get_contextname_collection(query:str,chat_history):
    
    '''
    Returns the appropriate collection name based on the user query.
    '''


    laboract_summary='''The provided documentation appears to be a subset of the Labour Act of Nepal, which outlines the rules and regulations governing labor laws in Nepal. The documentation covers various aspects of labor laws, including definitions, licenses for labor suppliers, obligations of labor suppliers and main employers, provisions for industry or service of special nature, inspection, supervision, and monitoring, liability of labor, prohibition on collecting fees from labor, powers of inspectors or employees, obligations of inspectors or employees, assistance to inspectors or employees, report submission, power of office to give direction, labor audit, making applications in convenient offices, performance evaluation, and miscellaneous provisions.
                        \n\nThe documentation also outlines the powers of the Labour Court, procedures for filing cases, and penalties for non-compliance. Additionally, it covers dispute resolution mechanisms, labour coordination, transitional provisions, and implementation of labour laws.\n\nSome of the key points include:\n\n* Definitions of key terms used in the Act\n* Licenses for labor suppliers, including the application process, fees, and scope of work\n* Obligations of labor suppliers, including payment of remuneration and benefits, provision of occupational safety and health measures, and reporting to the Department or Office\n* Obligations of main employers, including employment of labor through licensed labor suppliers, provision of remuneration and benefits, and regular information about labor suppliers' provision of remuneration and benefits\n* Provisions for industry or service of special nature, including tea-estate labor, construction labor, and transport labor\n* Inspection, supervision, and monitoring of labor suppliers, including the power to enter premises, inspect records, and take statements from employers, managerial level laborers, or other laborers\n* 
                        Liability of labor, including the responsibility of labor suppliers and main employers for payment of remuneration and benefits\n* Prohibition on collecting fees from labor, including the prohibition on labor suppliers collecting fees or commissions from laborers\n* Powers of inspectors or employees, including the power to enter premises, inspect records, and take statements from employers, managerial level laborers, or other laborers\n* Obligations of inspectors or employees, including the duty to perform duties without causing hindrance to the work of the enterprise or workplace and to keep confidential any information received during inspection\n* Assistance to inspectors or employees, including the duty of employers, laborers, and trade unions to render necessary assistance during inspections\n* Report submission, including the duty of inspectors or employees to submit a report to the Office within 15 days of completing an inspection\n* Power of office to give direction, including the power to give direction to make improvements or stop any act contrary to law based on the report submitted by the inspector or employee\n* Labour audit, including the duty of each enterprise to make a labour audit to ensure compliance with the Act, Regulation, and prevailing law\n* Making applications in convenient offices, including the right of laborers to make an application to the Office which is convenient to them\n* Performance evaluation, including the duty of employers to conduct performance evaluations of laborers and provide them with an opportunity to improve weaknesses\n* Miscellaneous provisions, including procedures for serving notices and processes, determination of basic remuneration, and special provisions for managers and managerial level laborers.
                        \n\nThe documentation also outlines various provisions related to labor laws in Nepal, covering aspects such as employment, working hours, leave, social security, and specific regulations for different industries. Some of the key points include:\n\n* Minimum standards for labor and employment contracts, ensuring that employees receive fair remuneration and benefits\n* Prohibitions, including forced labor, employment of children, and discrimination against labor based on religion, color, sex, caste, tribe, origin, language, or ideological conviction\n* Rights of labor, including the right to form and operate trade unions, seek remedy in case of infringement of rights, and enter into employment contracts that outline remuneration, benefits, and conditions of employment\n* Employment relationship, including the establishment of an employment relationship when an employment contract is entered into or when a labor is employed verbally or engaged in casual employment\n* Probation period and termination, including the right of an employer to enter into an employment contract with a labor that includes a probation period of up to six months\n* Holding labor in reserve and transfer, including the right of an employer to suspend work and hold a labor in reserve due to special circumstances\n* Formation of Labour Relation Committee, including the duty of an employer of an enterprise with ten or more labors to form a labor relation committee to discuss productivity increment, settle grievances, improve the working environment, and perform other prescribed functions\n* Provisions relating to employment, working hours, leave, and social security, including the types of employment, working hours, leave, and social security benefits afforded to employees in Nepal\n* Specific regulations for industries, including trainees and apprentices, construction labor, transport labor, and tourism labor\n\nOverall, the documentation provides a comprehensive framework for labor laws and regulations in Nepal, outlining the rights and obligations of laborers, employers, and labor suppliers, as well as the powers and procedures of inspectors and employees.'''

    passportact_summary='''The documentation provided is a subset of the acts and laws of Nepal, specifically the Passport Act, 2019 (2077). The act outlines the rules and regulations governing the issuance, use, and cancellation of passports in Nepal. Here is a detailed summary of the act:\n\n**Key Provisions:**\n\n1. **Passport Issuance:** The Department of Passports has the power to issue passports, which shall be done in accordance with international standards set by the International Civil Aviation Organization (ICAO). There are four types of passports: diplomatic, official, service, and ordinary.\n2. **Application and Issuance:** A Nepali citizen desiring a passport must submit an application to the Department, mission, or prescribed body, along with required documents. The Department may issue a passport in the name of a minor in certain conditions.\n3. **Cancellation of Passport:** The Department may cancel a passport in certain conditions, including if the passport holder informs the Department about a lost passport, if the passport is cancelled or revoked by a court order, or if the passport holder is no longer a Nepali citizen.\n4. **Use of Passport:** The provisions on the use of diplomatic and official passports shall be as prescribed. The holder of a diplomatic or official passport shall have to return the passport to the Department after the completion of the purpose.\n5. **Temporary and Special Condition Passports:** The Mission may issue a temporary passport to a Nepali citizen residing abroad if they submit an application for a new passport due to expiry, loss, destruction, or tampering of the previous passport.
                        \n6. **Disposal of Cancelled Passports:** The Department shall dispose of cancelled passports as prescribed.\n7. **Government as Plaintiff:** The Government of Nepal shall be a plaintiff in cases of offences punishable under this Act.\n8. **Appeal:** A person not satisfied with the decision of the Department for rejecting to issue a passport or cancelling a passport may file an appeal in the High Court of the province where the Department is situated within thirty-five days of receipt of information of such rejection or cancellation.\n9. **Delegation of Power:** The Department may delegate some of its powers to other bodies or officials as required.\n10. **Saving of Act Done in Good Faith:** No employee shall be personally liable for acts done in good faith under this Act.\n\n**Additional Provisions:**\n\n1. **Passport Issuance and Conditions:** The Department may issue a passport to a minor in certain conditions, such as a court order or with the consent of the guardian or curator.\n2. **Passport Cancellation and Revocation:** The Department may cancel a passport in certain conditions, such as if the passport holder informs the Department about a lost passport or if the passport is cancelled or revoked by a court order.\n3. **Travel Documents and Seamans Record Books:** The Department and mission may issue travel documents and seamans passports to Nepali citizens.\n4. **Offences and Punishment:** Any person who commits, causes to commit, or abets the commission of certain acts, such as submitting false details or using a passport for unauthorized purposes, shall be considered as committing an offence.
                        \n5. **Miscellaneous:** The colour, format, and size of passports shall be as prescribed. Fees shall be imposed for issuing ordinary passports, travel documents, seamans passports, and temporary passports, but not for diplomatic, official, or service passports.\n\nOverall, the documentation provides the legal framework for the issuance, cancellation, and revocation of passports, travel documents, and seamans record books in Nepal, as well as the offences and punishments related to these documents.'''

    election_summary='''The provided documentation is a subset of the acts and laws of Nepal, specifically related to the Election Commission of Nepal. The documentation outlines the rules, regulations, powers, and functions of the Election Commission in conducting free, fair, and transparent elections in Nepal.\n\nThe documentation can be broadly categorized into several sections, including:\n\n1. **Election Commission Act, 2017 (2073)**: This section outlines the rules and regulations governing the Election Commission of Nepal, including its powers, functions, and organizational structure.\n2. **Functions and Powers of the Commission**: This section details the Commission's powers and functions, including the power to use physical facilities, establish offices, appoint employees, and procure the services of experts.\n3. **Election Expenses and Budget Management**: This section deals with the management of election expenses, including the ceiling of election expenses, submission of expense details, and punishment for exceeding the specified ceiling.\n4. **Election Conduct and Employee Management**: This section covers various aspects of election conduct, including the power to make decisions on candidate disqualification, cancellation of elections, and employee management, including deputation of employees and procurement of goods and services.\n5. **Election Code of Conduct**: This section outlines the rules and regulations for conducting free, fair, and impartial elections in Nepal, including restrictions on the government, political parties, candidates, and media.
                                \n6. **Defense and Legal Proceedings**: This section deals with the Commission's power to defend its officials and employees in legal proceedings related to election-related activities.\n7. **Gender Friendliness and Inclusiveness**: This section requires the Commission to adopt gender-friendly and inclusive principles while preparing election-related policies, conducting programs, and deputing employees.\n8. **Power to Remove or Suspend**: This section grants the Commission the power to remove or recommend for suspension of government employees, security personnel, or teachers who commit acts prejudicial to the freedom, fairness, and impartiality of elections.\n\nSome key points from the documentation include:\n\n* The Election Commission has the power to use physical facilities, including those belonging to private organizations, for the purpose of conducting elections.\n* The Commission has the power to establish offices, appoint employees, and procure the services of experts to assist in the conduct of elections.\n* The Commission manages its own budget and has the power to collect financial resources for election purposes.\n* The Commission has the authority to make decisions on candidate disqualification and cancel elections if necessary.\n* The Commission has the power to defend its officials and employees in legal proceedings related to election-related activities.\n* The Commission is required to adopt gender-friendly and inclusive principles while preparing election-related policies, conducting programs, and deputing employees.\n\nOverall, the documentation provides a detailed framework for the conduct of elections in Nepal, outlining the powers, functions, and responsibilities of the Election Commission of Nepal.'''


    top_level_summaries_templates = [laboract_summary,passportact_summary,election_summary]

    collection_names=["TheLaborAct","PassportAct","ElectionCommissionAct"]

    #this below line should be run one time to create vectorstore once
    # summaryvector=Chroma.from_texts(texts=top_level_summaries_templates, embedding=embedding_model,persist_directory="/kaggle/working/summaryvectorstore")

    #retreiveing the persistent vectorstore from the directory
    summaryvectorstore=Chroma(persist_directory="./summaryvectorstore", embedding_function=embedding_model)

    summaryretriever= summaryvectorstore.as_retriever(search_type="similarity",search_kwargs={'k': 1})

        
    #normal prompt 
    prompt = ChatPromptTemplate.from_messages([
        ("system", """
     You are an expert at routing a user question to the appropriate data source collection.
    If user is trying to make normal conversation and greetings , just reply with one word "Greetings".
    If the user query is out of the summary, just reply with one word "Invalid".
    You answer should be strictly based on the summary provided and donot assume on your own.
    You should strictly reply with one word answer of the appropriate collection name from given list of collection names. You are strictly prohibited to generate introductory texts for the results.

    query={input}

    Retrieved summary: {context}

    collection names: {collections}

    Your answer:"""),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}")
    ])

    #rag chain without history
    chain = create_stuff_documents_chain(
        llm=llm,
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
        llm=llm,
        retriever=summaryretriever,
        prompt=prompt_with_history
    )


    #creating history aware retriever chain
    retrieval_chain = create_retrieval_chain(
        history_aware_retriever,
        chain
    )


    response=retrieval_chain.invoke({"input": query,
                      "collections":collection_names,
                    "chat_history":chat_history})
    
    return response["answer"]

