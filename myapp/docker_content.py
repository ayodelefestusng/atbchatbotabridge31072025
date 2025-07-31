# ==========================
# 🌐 Django & Project Settings
# ==========================
from django.conf import settings
from .models import Prompt

# ==========================
# 📦 Standard Library
# ==========================
import os
# import sys
# import uuid
import json
# import random
from datetime import datetime
from pprint import pprint

# ==========================
# 📦 Third-Party Core
# ==========================
from dotenv import load_dotenv
from PIL import Image
from pydantic import BaseModel, Field
from typing import Any, Dict, List, Optional, Literal
# from typing_extensions import TypedDict

# ==========================
# 🧠 Google Generative AI
# ==========================
import google.generativeai as genai
from google.generativeai import GenerativeModel, configure
from google.generativeai.types import HarmCategory, HarmBlockThreshold

# ==========================
# 🤖 LangChain Core & Community
# ==========================
# from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.messages import (
    AIMessage, HumanMessage, SystemMessage
)
# from langchain_core.documents import Document
# from langchain_core.output_parsers import JsonOutputParser
# from langchain_core.runnables import RunnableConfig
from langchain_core.vectorstores import InMemoryVectorStore

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.tools.sql_database.tool import QuerySQLDatabaseTool
# from langchain_community.vectorstores import Chroma
from langchain_community.utilities import SQLDatabase
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_tavily import TavilySearch

from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI

# ==========================
# 🔁 LangGraph Imports
# ==========================
from langgraph.graph import StateGraph, START, END, MessagesState
# from langgraph.graph.message import add_messages
# from langgraph.prebuilt import ToolNode, tools_condition
# from langgraph.checkpoint.memory import MemorySaver
# from langgraph_checkpoint.postgres import PostgresSaver
from langgraph.checkpoint.postgres import PostgresSaver

# from langgraph.errors import NodeInterrupt


# --- Project-Specific Imports ---

#AJADI-2


# Load .env file
load_dotenv()
# from langgraph.checkpoint.sqlite import SqliteSaver # <--- Updated import

# Retrieve variables
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
LANGSMITH_TRACING = os.getenv("LANGSMITH_TRACING")
LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY")
LANGSMITH_PROJECT = os.getenv("LANGSMITH_PROJECT")
LANGSMITH_ENDPOINT = os.getenv("LANGSMITH_ENDPOINT")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PDF_PATH = os.getenv("PDF_PATH", "default.pdf")


gemni = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-preview-04-17",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)

llm = gemni
model = llm # Consistent naming

embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
vector_store = InMemoryVectorStore(embeddings)

DB_URI = os.getenv("DB_URI")
db = SQLDatabase.from_uri(DB_URI)


class QueryOutput(BaseModel):
    query: str = Field(description="SQL query to run")
# Response schemas

# class Answer(BaseModel):
#     answerA: str
#     sentimentA: int
#     ticketA: list[str]
#     sourceA: list[str]


class Answer(BaseModel):
    answerA: str = Field(..., description="A clear, concise, empathetic, and polite response...")
    sentimentA: int = Field(..., description="An integer rating of the user's sentiment...")
    ticketA: list[str] = Field(..., description='A list of specific transaction or service channels...')
    sourceA: list[str] = Field(..., description='A list of specific sources...')


    
class Summary(BaseModel):
    """Conversation summary schema"""
    summaryS: str = Field(description="Summary of the entire conversation")
    sum_sentimentS: int = Field(description="Sentiment analysis of entire conversation")
    sum_ticketS: List[str] = Field(description="Channels with unresolved issues")
    sum_sourceS: List[str] = Field(description="All sources referenced in conversation")


class State(MessagesState):
    """State management for conversation flow"""
    question: str
    pdf_content: str
    web_content: str
    query_answerT: str
    answer: str
    sentiment: int
    ticket: List[str]
    source: List[str]
    attached_content:str
    summary : str
    sum_sentiment: int
    sum_ticket: List[str]
    sum_source: List[str]
    answerY: Answer
    metadatas: Dict[str, Any] = Field(default_factory=dict)
    summaryY: Summary


def get_time_based_greeting():
    """Return an appropriate greeting based on the current time."""
    current_hour = datetime.now().hour
    
    if 5 <= current_hour < 12:
        return "Good morning"
    elif 12 <= current_hour < 17:
        return "Good afternoon"
    elif 17 <= current_hour < 22:
        return "Good evening"
    else:
        return "Good night"

def get_current_datetime():
    """Returns the current date and time in YYYY-MM-DD HH:MM:SS format."""
    return datetime.now().strftime('%Y-%m-%d %H:%M:%S')

def safe_json(data):
    """Ensures safe JSON serialization to prevent errors."""
    try:
        return json.dumps(data)
    except (TypeError, ValueError):
        return json.dumps({})  # Returns an empty JSON object if serialization fails

# from langchain_ollama import OllamaEmbeddings

# embeddings = OllamaEmbeddings(model="llama3")
# from langchain_huggingface import HuggingFaceEmbeddings

# embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")


# embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")


global_vector_store = None
def initialize_vector_store():
    global global_vector_store
    if global_vector_store is None:
        # embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        embeddings = GoogleGenerativeAIEmbeddings( model="models/embedding-001",transport="rest")
        global_vector_store = InMemoryVectorStore(embedding=embeddings)
        file_path = os.path.join(settings.MEDIA_ROOT, 'pdfs', 'ATB Bank Nigeria Groq v2.pdf')
        if os.path.exists(file_path):
            loader = PyPDFLoader(file_path)
            docs = loader.load()
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, add_start_index=True)
            all_splits = text_splitter.split_documents(docs)
            global_vector_store.add_documents(documents=all_splits)
        else:
            print(f"Warning: PDF file not found at {file_path}. PDF retrieval tool will not work.")
# Call this function once at application startup
# initialize_vector_store()
initialize_vector_store()


# Define the tools
def search_pdf(state: State):  
    # This function would use global_vector_store
    if global_vector_store:
        ayula=state["messages"][-1].content
        attached_content=state["attached_content"]
        user_input = f"User Query:\n{ayula}\n\n:Attached File Content:\n{attached_content}"
        results = global_vector_store.similarity_search(user_input, k=3)
        
        # return "\n\n".join([doc.page_content for doc in results])
        return {"pdf_content":results}
    return "Error: Document knowledge base not initialized."

# pdf_retrieval_tool = Tool(
#     func=retrieve_answer,
#     name="bank_document_retrieval",
#     description="Useful for answering questions based on the XYZ Bank's internal knowledge base documents. Input should be a specific question."
# )
def search_web(state: State):  
        """Perform web search"""
        
        try:
            tavily_search = TavilySearch(max_results=2)
            ayula=state["messages"][-1].content
            attached_content=state["attached_content"]
            user_input = f"User Query:\n{ayula}\n\n:Attached File Content:\n{attached_content}"
           
            search = TavilySearch()
            search_docs = search.invoke(input=user_input)
            # search_docs = tavily_search.invoke(user_input)
            # print ("Web: Response Type:", search_docs)  # Debug print
            
            if any(error in str(search_docs) for error in ["ConnectionError", "HTTPSConnectionPool"]):
                return {"web_content": ""}
                
            formatted_docs = "\n\n---\n\n".join(
                f'<Document href="{doc["url"]}">\n{doc["content"]}\n</Document>'
                for doc in search_docs['results']
            )
            return {"web_content": formatted_docs}
        except Exception as e:
            print(f"Web search error: {e}")
            return {"web_content": ""}




def write_query(state: State)-> str:  
    # Ensure 'db' is properly initialized and accessible
    # Assuming 'db' is an instance of SQLDatabase from langchain_community.utilities
    global db
    
    dialect = db.dialect # Corrected from tuple
    top_k = 10
    table_info = db.get_table_info()
    current_time = get_current_datetime()


    sql_prompt = f"""
Given an input question, create a syntactically correct {dialect} query to
run to help find the answer. Unless the user specifies in his question a
specific number of examples they wish to obtain, always limit your query to
at most {top_k} results. You can order the results by a relevant column to
return the most interesting examples in the database.

Kindly note the current time: {current_time},
Never query for all the columns from a specific table, only ask for a the
few relevant columns given the question.

Pay attention to use only the column names that you can see in the schema
description. Be careful to not query for columns that do not exist. Also,
pay attention to which column is in which table.

Only use the following tables:
{table_info}
"""
    sys_msg = SystemMessage(content=sql_prompt)
    model_with_structure = llm.with_structured_output(QueryOutput) # Use llm here
    
    # Pass only the relevant part of the state, not the whole state for invoke
    ayula=state["messages"][-1].content
    attached_content=state["attached_content"]
    user_input = f"User Query:\n{ayula}\n\n:Attached File Content:\n{attached_content}"
    result = model_with_structure.invoke([sys_msg] + [HumanMessage(content=user_input)])
    
    execute_query_tool = QuerySQLDatabaseTool(db=db)
    resultT = execute_query_tool.invoke(result.query) # Access .query attribute
    
    prompt = (
        "Given the following user question, corresponding SQL query, "
        "and SQL result, answer the user question.\n\n"
        f'Question: {state["messages"][-1].content}\n'
        f'SQL Query: {result.query}\n' # Access .query attribute
        f'SQL Result: {resultT}'
    )
    try:
        responseY = llm.invoke(prompt)  # Use llm here
        print("\n--- Raw LLM Response Object (from write_query) ---",responseY.content) # Debug print
          # This should be a string or similar object
        return{"query_answerT": responseY.content}  # Return as a dictionary
        
    except Exception as e:
        print(f"Error invoking LLM: {e}")
        return "Error: Could not process the SQL query."    
    

    
    # Define the tools_condition logic
def tools_condition(state: State):
    last_message = state["messages"][-1]
    # This is a simplified condition. In a real scenario, you'd check if the
    # LLM explicitly requested a tool call.
    if last_message.tool_calls: # Assuming the LLM emits tool_calls
        return "tools"
    return "assistant" # Or "generate_answer" if that's a separate node


# This block should be outside the function or managed differently
    # to avoid re-initializing the workflow and memory on every call.
    # For demonstration, keeping it here, but it's inefficient.
    # It's better to build the graph once and then invoke it.
    
    # Example of how you would manage this in a proper application:
    # app = create_langgraph_app(DB_URI, tools, llm_with_tools) # a function that builds and compiles the graph
    # response = app.invoke({"messages": [HumanMessage(content=user_input)]}, {"configurable": {"session_id": session_id}})
# client = genai.Client(api_key=GOOGLE_API_KEY)
client = genai.configure(api_key=GOOGLE_API_KEY)
safety_settings = {"category": HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
                "threshold": HarmBlockThreshold.BLOCK_ONLY_HIGH}
                

# system_instruction = """
#             You are an expert software developer and a helpful coding assistant.
#             You are able to generate high-quality code in any programming language.
#         """
# config = types.GenerationConfig(
#             temperature=0.4,
#             top_p=0.95,
#             top_k=20,
#             candidate_count=1,
#             # seed=5,
#             max_output_tokens=100,
#             stop_sequences=["STOP!"],
#             # presence_penalty=0.0,
#             frequency_penalty=0.0,
#             # safety_settings=safety_settings,
#             # system_instruction=system_instruction,
#         )

def generate_answer(state: State):
        print("--- Inside assistant_node ---") # Debug print
        print("The real question:",  state["messages"][-1].content,) # Debug print
        # pdf_text = state["pdf_content"]
        pdf_text = "\n".join(doc.page_content for doc in state["pdf_content"])
        web_text = "\n".join(state["web_content"])
        # web_text = state["web_content"]
        query_answer=state["query_answerT"]
        attached_content=state["attached_content"]
        context = f"PDF Content:\n{pdf_text}\n\nWeb Content:\n{web_text}\n\nQuery Answer:\n{query_answer}\n\nQuery Answer:\n{attached_content}"
        greeting = get_time_based_greeting()
        # print ("AJADI",state)
        


        y = Prompt.objects.get(pk=1)  # Get the existing record
        retrieved_template1=y.response_prompt 
        response_prompt = retrieved_template1.format(
            greeting=get_time_based_greeting(),
            ayula=state["messages"][-1].content,
            attached_content=attached_content,  # Assuming no attached content for now
            context=context,
            pdf_text=pdf_text,
            web_text=web_text,
            query_answer=query_answer,
   
        )


        sys_msg = SystemMessage(content=response_prompt)

        model_with_structure = model.with_structured_output(Answer) 

        # print("--- Invoking LLM for structured output ---") # Debug print
        # response = [sys_msg] + state["messages"]

        # print("LLM Input Messages for structured output:")
        # for msg_item in response:
        #     print(f"    Type: {type(msg_item).__name__}, Content: {str(msg_item.content)[:200]}...")

        try:
            response = model_with_structure.invoke([sys_msg] + state["messages"])

            print("\n--- Raw LLM Response Object (from model_with_structured_output) ---")
            pprint(response) # This should be a Pydantic Answer object or similar
            print("--- End Raw LLM Response Object ---")

            ai_message_content = response.model_dump_json() # Convert Pydantic model to JSON string

            print("\n--- LLM Response as JSON String (after model_dump_json) ---")
            print(ai_message_content)
            print("--- End LLM Response as JSON String ---")

            # return {"messages": [AIMessage(content=ai_message_content, tool_calls=response.tool_calls if hasattr(response, 'tool_calls') else [])]}
            return {
                "answer": response.answerA,
                "sentiment": response.sentimentA,
                "ticket": response.ticketA,
                "source": response.sourceA,
            }

        except Exception as e:
            print(f"\n!!! ERROR during LLM invocation in assistant_node: {e}")
            # LangGraph expects a dict with "messages" key in a node's return
            return {"messages": [AIMessage(content=json.dumps({
                "answer": f"An error occurred during AI processing: {e}",
                "sentiment": -1,
                "source": [],
                "sourceA": ["Internal Error"]
            }))]}




def summarize_conversation(state: State):
        """Generate conversation summary"""
        
        ayulaTT=state["messages"]
        
        z = Prompt.objects.get(pk=1)  # Get the existing record
        retrieved_template2=z.summarize_prompt 
        summarize_prompt = retrieved_template2.format(
            ayulaTT=ayulaTT,   )

        try:
            model_with_structure = model.with_structured_output(Summary)
            response = model_with_structure.invoke([SystemMessage(content=summarize_prompt)]+state["messages"])
            print ("Summary Reo:",response)
            summary_data = {
                    "question": state["messages"][-1].content,
                    "answer": state['answer'],
                    "sentiment": state['sentiment'],
                    "ticket": state['ticket'],
                    "source": state['source'],
                    "attached_content": state['attached_content'],
                    "summary": response.summaryS,
                    "sum_sentiment": response.sum_sentimentS,
                    "sum_ticket": response.sum_ticketS,   # Unresolved issues
                    "sum_source": response.sum_sourceS
                }
            return { "metadatas": summary_data }
        except Exception as e:
            print(f"Error summarizing conversation: {e}")
            return { "metadatas": {"summary_data":"Unable to generate summary"}, }
        

def gambo(message,attached_content, session_id: str):
        
    print ("--- Building LangGraph workflow ---")
    """Builds the LangGraph workflow"""
    with PostgresSaver.from_conn_string(DB_URI) as memory:
        workflow = StateGraph(State)
        workflow.add_node("search_web", search_web)
        workflow.add_node("search_document", search_pdf)
        workflow.add_node("write_query", write_query)
        # workflow.add_node("generate_answer", generate_response)
        workflow.add_node("generate_answer", generate_answer)
        workflow.add_node("summarize", summarize_conversation)

        # Define workflow
        workflow.add_edge(START, "search_document")
        workflow.add_edge(START, "search_web")
        workflow.add_edge(START, "write_query")
        workflow.add_edge("search_document", "generate_answer")
        workflow.add_edge("search_web", "generate_answer")
        workflow.add_edge("write_query", "generate_answer")
        workflow.add_edge("generate_answer", "summarize")
        # workflow.add_edge("summarize", "final")
        workflow.add_edge("summarize", END)
        graph = workflow.compile(checkpointer=memory)
    


    # graph= gambo()  # Build the workflow graph
        config = {"configurable": {"thread_id": session_id}} # Changed to use session_id
       
        
        # message = HumanMessage(content=state["messages"][-1].content)
        # attached_file=state["attached_file"]
        output = graph.invoke({"messages": message,"attached_content": attached_content}, config)
        print("--- LangGraph workflow completed ---")
        return output
   

# Main function to process user messages
def process_message(message: str, session_id: str, file_path: str):
    """Main function to process user messages"""
    print("Processing message:")
    attached_content = ""

    # Only process image if file_path is provided
    if file_path:
        # ... (your existing image processing code) ...
        try:
            image = Image.open(file_path)
            image.thumbnail([512, 512]) # Resize for efficiency
            prompt = "Write out the content of the picture."
            configure(api_key=GOOGLE_API_KEY)
            modelT = GenerativeModel(model_name="gemini-2.0-flash",generation_config={"temperature": 0.7,"max_output_tokens": 512 })

            response = modelT.generate_content([image, prompt])

            attached_content = response.text
            print ("attached_content",attached_content)
        except Exception as e:
            print(f"Error processing image attachment: {e}")
            attached_content = f"Error: Could not process attached file ({e})"

    # Format user query with extracted file contents
    message =message 
    attached_content=attached_content
    user_input = f"User Query:\n{message}\n\nAttached File Content:\n{attached_content}"

    output= gambo(message,attached_content, session_id) # Invoke the graph with the input message and session ID
    # print("--- LangGraph workflow completed ---")
    # output = graph.invoke({"messages": [input_message]}, config)
    print("--- LangGraph workflow completed ---",output['answer'])
 
    return {
            "messages": output['answer'],
            "metadata": output.get("metadatas", {})
        }






