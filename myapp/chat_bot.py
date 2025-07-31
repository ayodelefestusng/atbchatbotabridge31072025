# # ==========================
# # 🌐 Django & Project Settings (Commented out as not used in standalone script)
# # ==========================
from django.conf import settings
from .models import Prompt,Prompt7

# ==========================
# 📦 Standard Library
# ==========================
import os
import json
from datetime import datetime
from pprint import pprint
import sqlite3
# ==========================
import pandas as pd


# ==========================
# 📦 Third-Party Core
# ==========================
from dotenv import load_dotenv
from PIL import Image
from pydantic import BaseModel, Field
from typing import Any, Dict, List, Optional, Literal

# ==========================
# 🧠 Google Generative AI
# ==========================
import google.generativeai as genai
from google.generativeai import GenerativeModel, configure
from google.generativeai.types import HarmCategory, HarmBlockThreshold

# # ==========================
# # 🤖 LangChain Core & Community
# # ==========================
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.tools import Tool # Explicitly import Tool
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.utilities import SQLDatabase
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_tavily import TavilySearch
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_groq import ChatGroq # For Groq LLM
from langchain_deepseek import ChatDeepSeek # Import ChatDeepSeek for DeepSeek LLM
from langchain_openai import ChatOpenAI

# # ==========================
# # 🔁 LangGraph Imports
# # ==========================
from langgraph.graph import StateGraph, START, END, MessagesState

from langgraph.prebuilt import ToolNode, tools_condition, create_react_agent
# from langgraph.checkpoint.sqlite import SqliteSaver # Using SqliteSaver as preferred
from django.conf import settings
from langgraph.checkpoint.postgres import PostgresSaver
# --- Project-Specific Imports ---
# AJADI-2


# Load .env file
load_dotenv()

# ==========================
# ⚙️ Configuration & Initialization
# ==========================
# Load API keys from environment variables for security
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY")
LANGSMITH_PROJECT = os.getenv("LANGSMITH_PROJECT")
LANGSMITH_ENDPOINT = os.getenv("LANGSMITH_ENDPOINT")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY") # Ensure this is set in .env if used
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") # Ensure this is set in .env if used
PDF_PATH = os.getenv("PDF_PATH", "default.pdf") # Default value for PDF_PATH

# # Set environment variables for LangSmith
os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_API_KEY"] = LANGSMITH_API_KEY if LANGSMITH_API_KEY else ""
os.environ["LANGSMITH_PROJECT"] = LANGSMITH_PROJECT if LANGSMITH_PROJECT else "Agent_Creation"
os.environ["LANGSMITH_ENDPOINT"] = LANGSMITH_ENDPOINT if LANGSMITH_ENDPOINT else "https://api.smith.langchain.com"
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY if GOOGLE_API_KEY else ""
os.environ["TAVILY_API_KEY"] = TAVILY_API_KEY if TAVILY_API_KEY else ""
os.environ["GROQ_API_KEY"] = GROQ_API_KEY if GROQ_API_KEY else ""


# Configure Google Generative AI
genai.configure(api_key=GOOGLE_API_KEY)
safety_settings = {
    "category": HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
    "threshold": HarmBlockThreshold.BLOCK_ONLY_HIGH
}


def safe_json(data):
    """Ensures safe JSON serialization to prevent errors."""
    try:
        return json.dumps(data)
    except (TypeError, ValueError):
        return json.dumps({})  # Returns an empty JSON object if serialization fails

# Initialize LLM
# User specified ChatGroq with deepseek model


py = Prompt7.objects.get(pk=1)  # Get the existing record
google_model = py.google_model

chatbot_model =py.chatbot_model


if chatbot_model =="gpt":
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, openai_api_key=OPENAI_API_KEY)
elif chatbot_model == "deepseek":
    llm = ChatDeepSeek(model="deepseek-chat", temperature=0, max_tokens=None, timeout=None, max_retries=2)
elif chatbot_model == "gemini":
    llm = ChatGoogleGenerativeAI(model=google_model, temperature=0, google_api_key=GOOGLE_API_KEY)    
elif chatbot_model == "groq":
    llm = ChatGroq(model="deepseek-r1-distill-llama-70b", temperature=0, max_tokens=None, timeout=None, max_retries=2)






# llms= ChatGroq( model="deepseek-r1-distill-llama-70b",temperature=0, max_tokens=None,timeout=None, max_retries=2,)
# llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0,google_api_key=GOOGLE_API_KEY)
# llm = ChatDeepSeek( model="deepseek-chat",  temperature=0, max_tokens=None, timeout=None,max_retries=2,)
# llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, openai_api_key=OPENAI_API_KEY)


# llm = init_chat_model("gpt-4o-mini", model_provider="openai")  # Removed because init_chat_model is not defined
# If you want to use OpenAI, you can use the following (make sure you have the correct import):





# If you want to use Gemini, uncomment the following and comment out ChatGroq
# llm= ChatGoogleGenerativeAI(
#     model="gemini-2.0-flash",
#     temperature=0,
#     max_tokens=None,
#     timeout=None,
#     max_retries=2,
# )


# llm = ChatGoogleGenerativeAI(
#     # model="gemini-2.5-flash-preview-04-17",
#     model ="gemini-2.5-flash",
#     temperature=0,
#     max_tokens=None,
#     timeout=None,
#     max_retries=2,
#     # other params...
# )



model = llm # Consistent naming

# Initialize Embeddings and Vector Store
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", transport="rest")
global_vector_store = None

def initialize_vector_store():
    """Initializes the vector store by loading and splitting the PDF document."""
    global global_vector_store
    if global_vector_store is None:
        global_vector_store = InMemoryVectorStore(embedding=embeddings)
        
        # Use the PDF_PATH from environment variables or default
        # file_path = PDF_PATH # This should be a full path or handled by Django settings
        file_path = os.path.join(settings.MEDIA_ROOT, 'pdfs', 'ATB Bank Nigeria Groq v2.pdf')
        
        # For local testing without Django settings, you might hardcode or derive it:
        # file_path = r"C:\Users\Pro\Desktop\Ayodele\25062025\myproject\media\pdfs\ATB Bank Nigeria Groq v2.pdf"

        if file_path and os.path.exists(file_path):
            try:
                loader = PyPDFLoader(file_path)
                docs = loader.load()
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, add_start_index=True)
                all_splits = text_splitter.split_documents(docs)
                global_vector_store.add_documents(documents=all_splits)
                print("PDF document loaded and processed successfully.")
            except Exception as e:
                print(f"Error loading PDF: {e}. PDF retrieval tool will not work.")
        else:
            print(f"Warning: PDF file not found at {file_path}. PDF retrieval tool will not work.")

# Call the initialization function
initialize_vector_store()

# Initialize SQLDatabase with the specified file path
# DB_FILE_PATH = r"C:\Users\Pro\Desktop\Ayodele\25062025\myproject\db.sqlite3"
# DB_URI = f"sqlite:///{DB_FILE_PATH}" 
DB_URI = os.getenv("DB_URI")
db = None
try:
    db = SQLDatabase.from_uri(DB_URI)
    print(f"SQLDatabase connected to {DB_URI} successfully.")
except Exception as e:
    print(f"Error connecting to SQLDatabase at {DB_URI}: {e}. SQL query tool will not be available.")


# ==========================
# 📝 Pydantic Schemas
# ==========================
# class Answer(BaseModel):
#     answerA: str = Field(..., description="A clear, concise, empathetic, and polite response...")
#     sentimentA: int = Field(..., description="An integer rating of the user's sentiment...")
#     ticketA: list[str] = Field(..., description='A list of specific transaction or service channels...')
#     sourceA: list[str] = Field(..., description='A list of specific sources...')



from pydantic import BaseModel, Field
from typing import List

class Answer(BaseModel):
    answerA: str = Field(description="Polite, empathetic response to the user")
    sentimentA: int = Field(description="Sentiment score from -2 to +2")
    ticketA: List[str] = Field(description="Relevant service channels")
    sourceA: List[str] = Field(description="Sources used to generate the answer")
    
class Summary(BaseModel):
    """Conversation summary schema"""
    summaryS: str = Field(description="Summary of the entire conversation")
    sum_sentimentS: int = Field(description="Sentiment analysis of entire conversation")
    sum_ticketS: List[str] = Field(description="Channels with unresolved issues")
    sum_sourceS: List[str] = Field(description="All sources referenced in conversation")

class PDFRetrievalInput(BaseModel):
    """Input schema for the pdf_retrieval_tool."""
    query: str = Field(description="The user's query to search for within the PDF document.")

class SQLQueryInput(BaseModel):
    """Input schema for the sql_query_tool."""
    query: str = Field(description="The natural language question to be converted into a SQL query and executed.")

# ==========================
# 📊 State Management
# ==========================
class State(MessagesState):
    """State management for conversation flow"""
    question: Optional[str] = None
    pdf_content: Optional[str] = None # Changed to Optional[str]
    web_content: Optional[str] = None
    query_answerT: Optional[str] = None
    answer: Optional[str] = None
    sentiment: Optional[int] = None
    ticket: Optional[List[str]] = None
    source: Optional[List[str]] = None
    attached_content: Optional[str] = None
    summary: Optional[str] = None
    sum_sentiment: Optional[int] = None
    sum_ticket: Optional[List[str]] = None
    sum_source: Optional[List[str]] = None
    answerY: Optional[Answer] = None
    metadatas: Optional[Dict[str, Any]] = None
    summaryY: Optional[Summary] = None


# ==========================
# 🛠️ Tools
# ==========================
def retrieve_from_pdf(query: str):
    """Performs a document query using the initialized vector store."""
    if global_vector_store:
        results = global_vector_store.similarity_search(query, k=3)
        # Return content as a single string
        return {"pdf_content": "\n\n".join([doc.page_content for doc in results])}
    return {"pdf_content": "Error: Document knowledge base not initialized."}


pdf_retrieval_tool = Tool(
    name="pdf_retrieval_tool",
    description=(
        "Useful for answering questions based on the bank's internal knowledge base documents. "
        "The input to this tool should be a specific question or a key phrase from the user's query."
    ),
    func=retrieve_from_pdf,
    args_schema=PDFRetrievalInput,
)

def search_web_func(query: str) -> str:
    """Performs web search and returns structured tool output."""
    try:
        tavily_search = TavilySearch(max_results=2)
        search_docs = tavily_search.invoke(query)

        if any(error in str(search_docs) for error in ["ConnectionError", "HTTPSConnectionPool"]):
            return "Web search connection error."
            
        formatted_docs = "\n\n---\n\n".join(
            f'<Document href="{doc["url"]}">\n{doc["content"]}\n</Document>'
            for doc in search_docs.get("results", [])
        )
        return formatted_docs
    except Exception as e:
        print(f"Web search error: {e}")
        return f"Error during web search: {e}"

tavily_search_tool = Tool(
    name="tavily_search_tool",
    description=(
        "Useful for answering general questions or questions requiring up-to-date information "
        "from the web. The input should be a concise search query."
    ),
    func=search_web_func,
    args_schema=SQLQueryInput, # Re-using SQLQueryInput schema for simplicity, but a dedicated WebSearchInput could be better
)


# Initialize SQL Toolkit and Agent globally if db is available
SQL_TOOLKIT = None
SQL_AGENT = None
if db:
    try:
        SQL_TOOLKIT = SQLDatabaseToolkit(db=db, llm=llm)
        sql_tools_list = SQL_TOOLKIT.get_tools() # Get tools specific to the SQL database

        SQL_SYSTEM_PROMPT = """
You are an agent designed to interact with a SQL database.
Given an input question, create a syntactically correct {dialect} query to run,
then look at the results of the query and return the answer. Unless the user
specifies a specific number of examples they wish to obtain, always limit your
query to at most {top_k} results.

You can order the results by a relevant column to return the most interesting
examples in the database. Never query for all the columns from a specific table,
only ask for the relevant columns given the question.

You MUST double check your query before executing it. If you get an error while
executing a query, rewrite the query and try again.

DO NOT make any DML statements (INSERT, UPDATE, DELETE, DROP etc.) to the
database.

To start you should ALWAYS look at the tables in the database to see what you
can query. Do NOT skip this step.

Then you should query the schema of the most relevant tables.
""".format(
            dialect=db.dialect,
            top_k=5,
        )

        SQL_AGENT = create_react_agent(
            llm,
            sql_tools_list, # Use the tools from the SQL toolkit
            prompt=SQL_SYSTEM_PROMPT,
        )
        print("SQL Agent initialized successfully.")
    except Exception as e:
        print(f"Error initializing SQL Agent: {e}. SQL query tool will not be available.")
        SQL_TOOLKIT = None
        SQL_AGENT = None


class QueryOutput(BaseModel):
    query: str = Field(description="SQL query to run")
def get_current_datetime():
    """Returns the current date and time in YYYY-MM-DD HH:MM:SS format."""
    return datetime.now().strftime('%Y-%m-%d %H:%M:%S')


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


# DB_URI = os.getenv("DB_URI")
# DB_URI = "db.sqlite3"
# DB_URI = "sqlite:///db.sqlite3"
DB_URI = os.getenv("DB_URI")
db = SQLDatabase.from_uri(DB_URI)

 # Ensure 'db' is properly initialized and accessible
    # Assuming 'db' is an instance of SQLDatabase from langchain_community.utilities

dialect = db.dialect # Corrected from tuple
top_k = 10
table_info = db.get_table_info()
current_time = get_current_datetime()
greetings = get_time_based_greeting() # Get greeting based on current time

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
model_with_structure1 = llm.with_structured_output(QueryOutput) # Use llm here

from langchain_community.tools.sql_database.tool import QuerySQLDatabaseTool
def execute_sql_query_func2(query: str) -> str:
    """Executes a SQL query using the pre-initialized SQL agent and returns the result."""
    result = model_with_structure1.invoke([sys_msg] + [HumanMessage(content=query)])
    execute_query_tool = QuerySQLDatabaseTool(db=db)
    resultT = execute_query_tool.invoke(result.query) # Access .query attribute
    prompt = (
        "Given the following user question, corresponding SQL query, "
        "and SQL result, answer the user question.\n\n"
        f'Question: {query}\n'
        f'SQL Query: {result.query}\n' # Access .query attribute
        f'SQL Result: {resultT}'
    )

    try:
        responseY = llm.invoke(prompt)  # Use llm here
        # print("\n--- Raw LLM Response Object (from write_query) ---",responseY.content) # Debug print
            # This should be a string or similar object
        return{ responseY.content}  # Return as a dictionary
    
    except Exception as e:
        return{"SQL Agent not initialized. Cannot execute query."}
   

def execute_sql_query_func(query: str) -> str:
    """Executes a SQL query using the pre-initialized SQL agent and returns the result."""
    if SQL_AGENT:
        try:
            response_generator = SQL_AGENT.stream(
                {"messages": [HumanMessage(content=query)]},
                stream_mode="values",
            )
            
            full_response_content = []
            for chunk in response_generator:
                if 'messages' in chunk and chunk['messages']:
                    last_message = chunk['messages'][-1]
                    if hasattr(last_message, 'content') and last_message.content:
                        full_response_content.append(last_message.content)
            
            return "\n".join(full_response_content) if full_response_content else "No response from SQL agent."
        except Exception as e:
            return f"Error executing SQL query: {e}"
    return "SQL Agent not initialized. Cannot execute query."

sql_query_tool = Tool(
    name="sql_query_tool",
    description=(
        "Useful for answering questions that require querying a SQL database. "
        "The input to this tool should be a natural language question about the database content, "
        "e.g., 'How many users are there?', 'List all products and their prices'. "
        "Use this tool ONLY if the question is clearly about structured data that would reside in a database."
    ),
    func=execute_sql_query_func2,
    args_schema=SQLQueryInput,
)

# Combine all tools
# tools = [tavily_search_tool, pdf_retrieval_tool,sql_query_tool]
tools = [pdf_retrieval_tool,sql_query_tool]
if SQL_AGENT: # Only add SQL tool if it was initialized successfully
    tools.append(sql_query_tool)

# ==========================
# 🧠 Graph Nodesayu


# ==========================
# This node will decide which tool to use or if to generate a final answer
def agent_node(state: State):
    print ("--- Inside agent_node ---")
    """
    The agent node decides whether to call a tool or generate a final answer.
    It binds tools to the LLM and invokes it with the current message history.
    """
    llm_with_tools = llm.bind_tools(tools)
    messages = state.get("messages", [])
    
    if not messages:

        return {"messages": [AIMessage(content="Hello! I am Damilola, your AI-powered virtual assistant. Welcome to ATB Bank. How can I help you today?")]}
      
      
    retrieved_template1=py.response_prompt 
    # response_prompt = retrieved_template1.format( greetings=greetings )
    response_prompt = retrieved_template1.format( greetings=greetings )
    system_prompt = SystemMessage(content=response_prompt)


    
#     system_prompt = SystemMessage(
#         content=(f""" 
#             "You are Damilola, the AI-powered virtual assistant for ATB Bank."
#             "Your primary goal is to answer questions using the tools you have access to. "
#             "Your core purpose is to deliver professional, accurate, and courteous customer support while performing data analytics when applicable."
#             "Always be empathetic, non-judgmental, and polite, ensuring every interaction reflects ATB Bank's commitment to exceptional service."


#             "You have the following tools available:\n"
#             "- `tavily_search_tool`: Use this to find general information on the web or current events. "
#             "  It takes a `query` (string) as input.\n"
#             "- `pdf_retrieval_tool`: Use this to look up information from the bank's internal documents. "
#             "  It takes a `query` (string) as input, which should be the user's question or key phrases related to bank services or requirements.\n"
#             "- `sql_query_tool`: Use this to query a SQL database for structured data, such as counts of items, lists of entities, etc. "
#             "  It takes a `query` (string) as input, which should be a natural language question about the database content. "
#             "  Only use this if the question is clearly about structured data that would reside in a database."
#             "  For example, if asked 'How many customers do we have?', use the `sql_query_tool`.\n\n"
#             "Always prefer using a tool if the user's question requires external knowledge or specific data. "
#             "You MUST use a tool to get the information first, and then formulate your final answer. "
#             "Do not answer questions from your internal knowledge base if a tool can be used. "
#             "Respond in a clear, polite, and helpful manner. Do NOT output structured JSON for your final answer here; simply provide plain text or tool calls."
#             1. Introduction and Tone:
#         •  Greeting: Always start by introducing yourself politely, tailored to the current time:{greetings} . For example: "Good [morning/afternoon/evening] and welcome to ATB Bank. I’m Damilola, your AI-powered virtual assistant and Data Analyst. How can I assist you today? 😊"
#         •  Language: Respond in the user's preferred language, matching the language of their message.
#         •  Politeness: Maintain a consistently polite and professional tone.
#         •  Emojis: Use emojis sparingly but appropriately to convey empathy and friendliness, matching the user's tone (e.g., 🥳, 🙂‍↕️, 😏, 😒, 🙂‍↔️).
#     2. Information Handling and Tools:
#     •  Commitment: Your responses must always indicate you are a member of ATB Bank (e.g., "we offer competitive loan rates," "our services include...").
#     3. Complaint and Issue Resolution:
#     •  Empathy: When responding to complaints, express genuine empathy and acknowledge the user's feelings. Use appropiate emojis to response to customer's feelings
#     •  Resolution Process: First, attempt to resolve the issue using information from PDF Content, Web Search, or SQL Database tools.
#     •  Unresolved Issues & Escalation: If you cannot resolve the issue or the user remains unsatisfied despite your efforts: 
#     o  Courteously inform the user that the issue will be escalated to the support team.
#     o  Categorize the unresolved issue by its relevant channel (e.g., POS, ATM, Web).
#     o  Communicate the action taken (e.g., "I understand your frustration. I'm escalating this to our dedicated support team for further investigation. They will reach out to you shortly regarding your ATM transaction issue.").
#     •  Resolution Update: Clearly communicate the actions taken or the resolution achieved for an issue.
#     4. Customer Engagement and Closing:
#     •  Positive Feedback: Thank customers for their kind words or positive feedback.
#     •  Apology: Sincerely apologize for any dissatisfaction or inconvenience caused.
#     •  Closing: End every interaction politely by asking if the user needs further assistance. For example: "Is there anything else I can assist you with today? I'm here to help! 😊

# """
#         )
#     )



    
    full_messages = [system_prompt] + messages
    
    response = llm_with_tools.invoke(full_messages)
     # The response from llm_with_tools.invoke will either be an AIMessage with content
    # or an AIMessage with tool_calls.
    return {"messages": [response]}

# Utility function for greeting (from original code)


def generate_final_answer_node(state: State):
    """
    This node generates the final answer based on the accumulated context from tools.
    It ensures the output is a plain AIMessage.
    """
    print("--- Inside generate_final_answer_node2 ---")
    
    # Extract relevant content from the state
    # Ensure user_query is extracted from the last HumanMessage in the history
    user_query = ""
    for msg in reversed(state.get("messages", [])):
        if isinstance(msg, HumanMessage):
            user_query = msg.content
            break

    # Get tool outputs from state, ensuring they are strings
    pdf_text_output = state.get("pdf_content", "")
    web_text_output = state.get("web_content", "")
    query_answer_output = state.get("query_answerT", "")
    attached_content = state.get("attached_content", "None")

    context_parts = []
    if pdf_text_output:
        context_parts.append(f"PDF Content:\n{pdf_text_output}")
    if web_text_output:
        context_parts.append(f"Web Content:\n{web_text_output}")
    if query_answer_output:
        context_parts.append(f"Query Answer:\n{query_answer_output}")
    if attached_content and attached_content != "None":
        context_parts.append(f"Attached Content:\n{attached_content}")

    context = "\n\n".join(context_parts) if context_parts else "No additional context found."

    greeting = get_time_based_greeting() # Using the utility function

    # Hardcoded prompt template for final answer (plain text output)
    # Removed the structured output instructions from this prompt
    response_prompt = f"""
You are Damilola, the AI-powered virtual assistant for ATB Bank.
Your core purpose is to deliver professional, accurate, and courteous customer support.
When required used tool ;{tools}

{greeting}!

Based on the following information, please provide a clear, concise, empathetic, and polite answer to the user's question.
User Question: {user_query}

Context from tools:
{context}

Please ensure your response is in plain text, directly addressing the user's question.
Always maintain a polite and professional tone, and conclude by asking if there's anything else you can assist with today.
"""
    
    print("--- PROMPT for final answer ---")
    print(response_prompt)

    try:
        # Pass both the system prompt and the user's last message for context
        messages_for_llm = [SystemMessage(content=response_prompt)]
        if user_query: # Only add user message if it's not empty
            messages_for_llm.append(HumanMessage(content=user_query)) 
        
        final_ai_response = llm.invoke(messages_for_llm)
        
        # Update the state with the final answer and related metadata
        # The 'messages' list in the state should be updated with the new AIMessage
        new_messages = state.get("messages", []) + [AIMessage(content=final_ai_response.content)]

        return {
            "answer": final_ai_response.content,
            "sentiment": 1, # Placeholder, could be derived by another LLM call
            "ticket": [], # Placeholder
            "source": ["Internal Knowledge" if not context_parts else "Mixed Sources"], # Placeholder
            "messages": new_messages # Update messages in state
        }

    except Exception as e:
        print(f"\n!!! ERROR during final answer generation: {e}")
        # Ensure 'messages' is always a list of BaseMessage objects
        error_message_content = f"An error occurred while generating the final answer: {e}"
        new_messages = state.get("messages", []) + [AIMessage(content=error_message_content)]
        return {
            "answer": error_message_content,
            "sentiment": -1,
            "ticket": [],
            "source": ["Internal Error"],
            "messages": new_messages
        }

def summarize_conversation(state: State):
    """Generate conversation summary"""
    print("--- Inside summarize_conversation node ---")
    messages = state.get("messages", [])
    
    # Check if messages is empty or invalid before proceeding
    if not messages:
        print("Warning: No messages found for summarization.")
        return { "metadatas": {"summary_data":"Unable to generate summary - no messages"}, }

    # Convert messages to a readable format for summarization
    conversation_history = "\n".join([f"{msg.type}: {msg.content}" for msg in messages if hasattr(msg, 'content') and msg.content is not None])

    # Hardcoded summarize prompt template for structured summary output
    summarize_prompt_template = """
Please summarize the following conversation. Also, identify the overall sentiment of the conversation (e.g., 1 for positive, 0 for neutral, -1 for negative), any unresolved issues/channels, and all sources referenced.

Conversation History:
{conversation_history}

Provide the summary, sentiment, unresolved tickets, and sources in the following JSON format:
{{
    "summaryS": "Summary of the entire conversation",
    "sum_sentimentS": "Sentiment analysis of entire conversation (1, 0, or -1)",
    "sum_ticketS": ["Channels with unresolved issues"],
    "sum_sourceS": ["All sources referenced in conversation"]
}}
"""
    summarize_prompt = summarize_prompt_template.format(conversation_history=conversation_history)

    try:
        model_with_structure = model.with_structured_output(Summary)
        # Pass the system prompt and the full message history
        # Ensure messages are not empty before passing to invoke
        messages_for_summary = [SystemMessage(content=summarize_prompt)] + messages
        response = model_with_structure.invoke(messages_for_summary)
        print ("Summary Response:", response)
        
        # Access attributes directly, as 'response' is a Pydantic object
        summary_data = {
            "question": state.get("messages", [])[-1].content if state.get("messages") else "",
            "answer": state.get('answer', ''),
            "sentiment": state.get('sentiment', 0),
            "ticket": state.get('ticket', []),
            "source": state.get('source', []),
            "attached_content": state.get('attached_content', ''),
            "summary": response.summaryS, # Direct attribute access
            "sum_sentiment": response.sum_sentimentS, # Direct attribute access
            "sum_ticket": response.sum_ticketS, # Direct attribute access
            "sum_source": response.sum_sourceS # Direct attribute access
        }
    
        return { "metadatas": summary_data }
    
    except Exception as e:
        print(f"Error summarizing conversation: {e}")
        return { "metadatas": {"summary_data":f"Unable to generate summary: {e}"}, }




# Define the condition to check if a tool was called
# def should_continue(state: State) -> str:
#     """
#     Determines whether the agent should continue by calling a tool or
#     generate a final answer.
#     """
#     last_message = state.get("messages", [])[-1]
#     if last_message.tool_calls:
#         return "tools"
#     # If no tool calls, it means the LLM is ready to generate a final answer
#     return "generate_final_answer"


# Define the condition to check if a tool was called
def should_continue(state: State) -> str:
    """
    Determines whether the agent should continue by calling a tool or
    generate a final answer.
    """
    last_message = state.get("messages", [])[-1]
    if last_message.tool_calls:
        return "tools"
    # If no tool calls, it means the LLM is ready to generate a final answer
    return "generate_final_answer"


# ==========================
# 🔄 Graph Workflow
# ==========================


def gambo(message: HumanMessage, attached_content: str, session_id: str):
    print ("--- Building LangGraph workflow ---")
    with PostgresSaver.from_conn_string(DB_URI) as memory:

    # # Use the specified DB_FILE_PATH for SqliteSaver
    # conn_checkpoint = None
    # try:
    #     conn_checkpoint = sqlite3.connect(DB_FILE_PATH, check_same_thread=False)
    #     memory = SqliteSaver(conn=conn_checkpoint)
    #     print("SQLite checkpointing connected successfully.")
    # except Exception as e:
    #     print(f"Error connecting to SQLite for checkpointing: {e}. Checkpointing will not be available.")
    #     from langgraph.checkpoint.memory import InMemorySaver
    #     memory = InMemorySaver()

        workflow = StateGraph(State)
        workflow.add_node("agent_node", agent_node) # Node for LLM decision-making (tool or answer)
        workflow.add_node("tools", ToolNode(tools=tools)) # Node for executing tools
        workflow.add_node("generate_final_answer", generate_final_answer_node) # Node for generating final text response
        workflow.add_node("summarize", summarize_conversation) # Node for summarizing

        # Define workflow edges
        workflow.add_edge(START, "agent_node")
        
        # The agent_node decides if it needs to use a tool or generate a final answer
        workflow.add_conditional_edges(
            "agent_node",
            should_continue,
            {
                "tools": "tools",
                "generate_final_answer": "generate_final_answer",
            },
        )
        
        # After a tool is executed, return to the agent_node to decide the next step
        workflow.add_edge("tools", "agent_node")
        
        # After generating the final answer, proceed to summarize
        workflow.add_edge("generate_final_answer", "summarize")
        
        # After summarizing, the workflow ends
        workflow.add_edge("summarize", END)

        graph = workflow.compile(checkpointer=memory)
        
        config = {"configurable": {"thread_id": session_id}}
        
        # Initial state for the graph invocation
        initial_state = {"messages": [message], "attached_content": attached_content}
        
        output = graph.invoke(initial_state, config)
        print("--- LangGraph workflow completed ---")
        
        # Close the checkpointing connection after invocation
        # if conn_checkpoint:
        #     conn_checkpoint.close()

        return output
        

# Main function to process user messages
def process_message(message_content: str, session_id: str, file_path: Optional[str] = None):
    """Main function to process user messages"""
    print("Processing message:")
    attached_content = "None"

    # Only process image if file_path is provided
    if file_path and os.path.exists(file_path):
        try:
            image = Image.open(file_path)
            image.thumbnail([512, 512]) # Resize for efficiency
            prompt = "Describe the content of the picture in detail."
            configure(api_key=GOOGLE_API_KEY)
            modelT = GenerativeModel(model_name="gemini-2.0-flash", generation_config={"temperature": 0.7,"max_output_tokens": 512 })

            response = modelT.generate_content([image, prompt])
            attached_content = response.text
            print ("Attached content from image:", attached_content)
        except Exception as e:
            print(f"Error processing image attachment: {e}")
            attached_content = f"Error: Could not process attached file ({e})"
    elif file_path:
        print(f"Warning: Attached file not found at {file_path}. Skipping image processing.")

    # Create HumanMessage for the graph
    user_message_for_graph = HumanMessage(content=message_content)

    output = gambo(user_message_for_graph, attached_content, session_id)
    
    print("--- LangGraph workflow completed --- Final Output:", output)
    
    # Extract the final answer content
    final_answer_content = output.get('answer', 'No answer generated.')
    
    return {
        "messages": final_answer_content,
        "metadata": output.get("metadatas", {})
    }

# # ==========================
# # ▶️ Main Execution
# # ==========================
# if __name__ == "__main__":

#     # Function to set up a sample SQLite database
#     def setup_sample_db(db_file_path):
#         conn_local = None
#         try:
#             # Ensure the directory exists
#             os.makedirs(os.path.dirname(db_file_path), exist_ok=True)
#             # Connect to the database file
#             conn_local = sqlite3.connect(db_file_path)
#             cursor = conn_local.cursor()

#             # Create a sample table
#             cursor.execute("""
#                 CREATE TABLE IF NOT EXISTS customers (
#                     id INTEGER PRIMARY KEY AUTOINCREMENT,
#                     name TEXT NOT NULL,
#                     email TEXT UNIQUE NOT NULL,
#                     age INTEGER,
#                     city TEXT
#                 );
#             """)

#             # Insert some sample data (only if not already present)
#             cursor.execute("INSERT OR IGNORE INTO customers (id, name, email, age, city) VALUES (1, 'Alice Smith', 'alice@example.com', 30, 'New York');")
#             cursor.execute("INSERT OR IGNORE INTO customers (id, name, email, age, city) VALUES (2, 'Bob Johnson', 'bob@example.com', 24, 'Los Angeles');")
#             cursor.execute("INSERT OR IGNORE INTO customers (id, name, email, age, city) VALUES (3, 'Charlie Brown', 'charlie@example.com', 35, 'New York');")
#             cursor.execute("INSERT OR IGNORE INTO customers (id, name, email, age, city) VALUES (4, 'Diana Prince', 'diana@example.com', 29, 'London');")
            
#             conn_local.commit()
#             print(f"Sample 'customers' table created and populated in {db_file_path}.")
#         except Exception as e:
#             print(f"Error setting up sample database: {e}")
#         finally:
#             if conn_local:
#                 conn_local.close()

#     # Setup the sample database before running the agent
#     setup_sample_db(DB_FILE_PATH)

#     # Example user queries
#     # user_query = "What are the bank's account opening requirements and which services are available?"
#     user_query = "How many customers do we have?" # Example query for SQL tool
#     # user_query = "What is the capital of France?" # Example query for web search tool

#     session_id = "test_session_1"
    
#     # No file attached for this example
#     attached_file_path = None 

#     try:
#         final_output = process_message(user_query, session_id, attached_file_path)
        
#         print("\n==================\nFinal Answer:")
#         print(final_output['messages'])
#         print("\nMetadata (Summary):")
#         pprint(final_output['metadata'])

#     except Exception as e:
#         print(f"\nAn error occurred during execution: {e}")









importants= """
gross
net
charge
amount
Basic
Transport
Housing
NHF
NHIS
NSITF
tax
pension
employerPension
deduction
OtherAllowance
_id
fullname
id
employeeID.phone
employeeID.accountNumber
employeeID.pencomID
employeeID.annualSalary
employeeID.jobRole.name
employeeID.accountName
meta.annualGross
meta.sumBasicHousingTransport
meta.earnedIncome
meta.earnedIncomeAfterRelief
meta.sumRelief
"""

def important ():
    key_fields = [line.strip() for line in importants.strip().splitlines()]
    return key_fields


desired_columnsT2= [
     "type", "payment_status", "gross", "net", "charge", "amount", "_id", "Basic", "Housing", "Transport",
    "Leave", "Medical", "Bread", "Shoe", "Car", "House", "Nepa", "allOtherItems", "notPayableDetails",
    "Leave_schedule", "Medical_schedule", "Bread_schedule", "Shoe_schedule", "Car_schedule", "Nepa_schedule",
    "Basic_type", "Transport_type", "Housing_type", "Leave_type", "Medical_type", "Bread_type", "Shoe_type",
    "Car_type", "House_type", "Nepa_type", "reliefs", "NHF", "NHIS", "NSITF", "benefits", "benefit", "tax",
    "pension", "employerPension", "bonuses", "bonus", "deduction", "OtherAllowance", "allowance", "year",
    "paymentDate", "percentage", "month", "companyID", "CashAdvance", "leaveAllowance", "cashReimbursement",
    "__v", "createdAt", "updatedAt", "fullname", "id", "employeeID.companyLicense.assignedAt",
    "employeeID.companyLicense.licenseId", "employeeID.jobRole.jobRoleId", "employeeID.jobRole.name",
    "employeeID.employeeConfirmation.confirmed", "employeeID.employeeConfirmation.status",
    "employeeID.employeeConfirmation.date", "employeeID.termination.prorated.status",
    "employeeID.termination.status", "employeeID.termination.medicalBalance", "employeeID.termination.leaveBalance",
    "employeeID.termination.isPaid", "employeeID.termination.otherAllowancesBalance", "employeeID.termination.date",
    "employeeID.termination.reason", "employeeID.employeeType", "employeeID.employeeSubordinates",
    "employeeID.mentees", "employeeID.aboutEmployee", "employeeID.facebookUrl", "employeeID.twitterUrl",
    "employeeID.linkedInUrl", "employeeID.employeeTribe", "employeeID.allowanceType", "employeeID.annualSalary",
    "employeeID.costOfHire", "employeeID.hourlyRate", "employeeID.isActive", "employeeID.isConfirmed",
    "employeeID.dependents", "employeeID._id", "employeeID.employeeEmail", "employeeID.firstName",
    "employeeID.middleName", "employeeID.lastName", "employeeID.religion", "employeeID.gender", "employeeID.bankName",
    "employeeID.phone", "employeeID.accountNumber", "employeeID.salaryScheme._id", "employeeID.salaryScheme.name",
    "employeeID.salaryScheme.items", "employeeID.salaryScheme.country", "employeeID.salaryScheme.companyID",
    "employeeID.salaryScheme.employeeContribution", "employeeID.salaryScheme.employerContribution",
    "employeeID.salaryScheme.createdAt", "employeeID.salaryScheme.updatedAt", "employeeID.salaryScheme.__v",
    "employeeID.branchID._id", "employeeID.branchID.branchKeyName", "employeeID.branchID.branchName",
    "employeeID.branchID.companyID", "employeeID.branchID.createdOn", "employeeID.branchID.modifiedOn",
    "employeeID.branchID.createdAt", "employeeID.branchID.updatedAt", "employeeID.branchID.id",
    "employeeID.branchID.__v", "employeeID.staffID", "employeeID.employeeHireDate", "employeeID.dateOfBirth",
    "employeeID.employeeCadreStep", "employeeID.employeeCadre", "employeeID.createdOn", "employeeID.modifiedOn",
    "employeeID.companyID", "employeeID.companyFID", "employeeID.createdAt", "employeeID.updatedAt", "employeeID.id",
    "employeeID.__v", "employeeID.dailyPay", "employeeID.employeeManager", "employeeID.myx3ID", "employeeID.competency",
    "employeeID.departmentID", "employeeID.employementType", "employeeID.workArrangement", "employeeID.teamID",
    "employeeID.divisionID", "employeeID.accountName", "employeeID.bankCode", "employeeID.recipientCode",
    "employeeID.talentNominations", "employeeID.addonLicenses", "employeeID.leaveCategory", "employeeID.city",
    "employeeID.payRateType", "employeeID.businessUnitID", "employeeID.employeeCategory", "employeeID.lastHireDate",
    "employeeID.maritalStatus", "meta.annualGross", "meta.sumBasicHousingTransport", "meta.earnedIncome",
    "meta.earnedIncomeAfterRelief", "meta.sumRelief", "pfa", "taxAuthority", "employeeID.employeeConfirmation.processId",
    "employeeID.pencomID", "employeeID.profileImgUrl", "employeeID.nhfPIN", "employeeID.pfa", "employeeID.taxAuthority",
    "employeeID.taxID", "employeeID.promotionDate", "employeeID.bankCountry", "employeeID.branchCode",
    "employeeID.employeeTitle", "payslipPDFView.person.profileImgUrl", "payroll_id", "employee_ID", "Other Items"
]

desired_columnsT= [
     "type", "payment_status", "gross", "net", "charge", "amount", "_id", "Basic", "Housing", "Transport",
    "Leave", "Medical", "Bread", "Shoe", "Car", "House", "Nepa", "allOtherItems", "notPayableDetails",
    "Leave_schedule", "Medical_schedule", "Bread_schedule", "Shoe_schedule", "Car_schedule", "Nepa_schedule",
    "Basic_type", "Transport_type", "Housing_type", "Leave_type", "Medical_type", "Bread_type", "Shoe_type",
    "Car_type", "House_type", "Nepa_type", "reliefs", "NHF", "NHIS", "NSITF", "benefits", "benefit", "tax",
    "pension", "employerPension", "bonuses", "bonus", "deduction", "OtherAllowance", "allowance", "year",
    "paymentDate", "percentage", "month", "companyID", "CashAdvance", "leaveAllowance", "cashReimbursement",
    "__v", "createdAt", "updatedAt", "fullname", "id", "employeeID.companyLicense.assignedAt",
    "employeeID.companyLicense.licenseId", "employeeID.jobRole.jobRoleId", "employeeID.jobRole.name",
    "employeeID.employeeConfirmation.confirmed", "employeeID.employeeConfirmation.status",
    "employeeID.employeeConfirmation.date", "employeeID.termination.prorated.status",
    "employeeID.termination.status", "employeeID.termination.medicalBalance", "employeeID.termination.leaveBalance",
    "employeeID.termination.isPaid", "employeeID.termination.otherAllowancesBalance", "employeeID.termination.date",
    "employeeID.termination.reason", "employeeID.employeeType", "employeeID.employeeSubordinates",
    "employeeID.mentees", "employeeID.aboutEmployee", "employeeID.facebookUrl", "employeeID.twitterUrl",
    "employeeID.linkedInUrl", "employeeID.employeeTribe", "employeeID.allowanceType", "employeeID.annualSalary",
    "employeeID.costOfHire", "employeeID.hourlyRate", "employeeID.isActive", "employeeID.isConfirmed",
    "employeeID.dependents", "employeeID._id", "employeeID.employeeEmail", "employeeID.firstName",
    "employeeID.middleName", "employeeID.lastName", "employeeID.religion", "employeeID.gender", "employeeID.bankName",
    "employeeID.phone", "employeeID.accountNumber", "employeeID.salaryScheme._id", "employeeID.salaryScheme.name",
    "employeeID.salaryScheme.items", "employeeID.salaryScheme.country", "employeeID.salaryScheme.companyID",
    "employeeID.salaryScheme.employeeContribution", "employeeID.salaryScheme.employerContribution",
    "employeeID.salaryScheme.createdAt", "employeeID.salaryScheme.updatedAt", "employeeID.salaryScheme.__v",
    "employeeID.branchID._id", "employeeID.branchID.branchKeyName", 
    "employeeID.branchID.companyID", "employeeID.branchID.createdOn", "employeeID.branchID.modifiedOn",
    "employeeID.branchID.createdAt", 
    "employeeID.branchID.__v", "employeeID.staffID", "employeeID.employeeHireDate", "employeeID.dateOfBirth",
    "employeeID.employeeCadreStep", "employeeID.employeeCadre", "employeeID.createdOn",
    "employeeID.companyID", "employeeID.companyFID", "employeeID.createdAt", "employeeID.updatedAt", "employeeID.id",
    "employeeID.__v","employeeID.employeeManager", "employeeID.myx3ID", 
    "employeeID.departmentID", "employeeID.employementType", 
     "employeeID.city",
    "employeeID.payRateType", "employeeID.businessUnitID", "employeeID.employeeCategory", "employeeID.lastHireDate",
    "meta.annualGross", "meta.sumBasicHousingTransport",

]



def desire():
    desire = desired_columnsT
    return desire 

# Define your multiply function
def atb1(a, b):
    data1_raw = a
    data_raw = b
    systemprompt = f"""
You are a professional financial analyst specializing in payroll and variance analysis. Your task is to perform a detailed payroll comparison between two datasets I will provide: a "previous period":{data1_raw} dataset and a "current period":{data_raw} dataset.
Your analysis must follow these steps:
Identify Employee Status: For every employee ID across both datasets, determine their status as one of the following:
Continuing: Appears in both datasets.
New: Appears only in the current period dataset.
Departed: Appears only in the previous period dataset.
Calculate Variances: Compute the monetary variance (NGN) for Gross Pay, Tax, and Pension for each employee and for the overall totals.
Identify Key Drivers: Analyze the variances to find the main reasons for the changes. Specifically look for:
Changes in headcount (new hires vs. departures).
Pay raises or decreases for continuing employees.
Unusual changes, such as a change in a deduction (like tax) without a corresponding change in gross pay. This is a critical insight to identify.
You must structure your output as a professional report using Markdown formatting with the following exact sections:
1. Executive Summary:
Start with the single most important number: the total variance in Gross Pay.
State whether this variance is favorable (a cost decrease) or unfavorable (a cost increase) from the company's perspective.
Briefly state the primary reason for this variance (e.g., "driven by headcount changes").
2. Overall Payroll Summary:
Create a summary table comparing the totals of the two periods.
The table columns must be: Metric, Previous Period, Current Period, Variance (NGN), and Variance (%).
Include rows for Gross Pay, Total Tax, and Total Pension.
3. Detailed Variance Analysis:
Create a sub-section titled 3.1. Headcount Changes that lists the departed and new employees and the gross pay impact of each group.
Create a sub-section titled 3.2. Variances for Continuing Employees that explicitly calls out any employees with changes in pay or deductions, specifying the exact variance amount.
4. Reconciliation of Gross Pay Variance:
Provide a simple table that clearly shows how the individual key drivers (e.g., Departures, New Hires, Pay Raises) sum up to the total Gross Pay variance. This proves your analysis is correct.
5. Conclusion & Recommendations:
Conclude with clear, actionable recommendations based on your findings. For example: "Verify the authorization for [Employee]'s pay raise" or "Investigate the reason for the tax change for [Employee], as their gross pay was unchanged."
Ensure the tone is professional, objective, and data-driven. Use currency formatting (e.g., N5,200) throughout the report."""
    responseY = llm.invoke([
        systemprompt,
        HumanMessage(content="Please review")
    ])
    return responseY.content



# retrieved_template1 = py.variance_prompt
# systemprompt=retrieved_template1

def atb(old, new,llmv,retrieved_template6):
    old = old
    new = new
    systemprompt = f"""

You are a professional financial analyst specializing in payroll and variance analysis. Your task is to perform a detailed payroll comparison between two datasets provided in JSON format:

"Previous Payroll Period": {old}

"Current Payroll Period": {new}

Your analysis must be meticulous, data-driven, and presented in a clear, professional Markdown report.

Analysis Instructions

You must follow these steps precisely:

1. Identify Employee Status:
Use employeeID._id as the unique identifier for each employee across both datasets. Classify each employee into one of the following categories:

Continuing: The employeeID._id exists in both the previous and current datasets.

New: The employeeID._id exists only in the current dataset.

Departed: The employeeID._id exists only in the previous dataset.

Suspicious: Flag any continuing employee as "Suspicious" if ANY of the following conditions are met. Comparison must be exact and case-sensitive.

fullname has changed.

employeeID.bankName has changed (e.g., "Zenith Bank" vs. "Zenith Banks").

employeeID.accountNumber has changed.

employeeID.phone has changed.

Suspicious (Duplicate ID): If an employeeID._id appears more than once within the payslips array of the Current Payroll Period, it must be flagged as a critical data integrity issue.

Note: If a file contains multiple top-level payroll objects, consolidate all payslips into a single list for each period before starting the analysis.

2. Calculate Monetary Variances:
For each employee and for the overall totals, compute the monetary variance (Current - Previous) in NGN for the following fields:

gross (Gross Pay)

tax (Tax)

pension (Employee Pension Contribution)

3. Identify Key Drivers of Variance:
Analyze the data to determine the root causes of any financial changes. Your analysis must explicitly connect variances to:

Headcount Changes: The financial impact of new hires and departures.

Pay Changes: Changes in gross pay for continuing employees.

Anomalies & Data Quality: The financial impact of suspicious records, especially duplicate entries.

Required Output Format (Markdown)

Generate the report using the exact structure and formatting below.

1. Executive Summary

Start with a headline figure: the total variance in Gross Pay.

State whether the variance is favorable (cost decrease) or unfavorable (cost increase).

Briefly summarize the primary drivers (e.g., headcount changes, significant pay adjustments, data anomalies).

2. Overall Payroll Summary

Provide a Markdown table comparing the aggregate values:

Generated markdown
| Metric        | Previous Period | Current Period | Variance (NGN) | Variance (%) |
|---------------|-----------------|----------------|----------------|--------------|
| Gross Pay     |                 |                |                |              |
| Total Tax     |                 |                |                |              |
| Total Pension |                 |                |                |              |

3. Detailed Variance Analysis
3.1 Headcount Changes

List new and departed employees and their financial impact.

Departed Employees (Count)

Generated markdown
| Employee Name | Employee ID | Gross Pay Impact (NGN) |
|---------------|-------------|------------------------|
| ...           |             |                        |
| **Total Departures** | | **(Total Value)** |
IGNORE_WHEN_COPYING_START
content_copy
download
Use code with caution.
Markdown
IGNORE_WHEN_COPYING_END

New Employees (Count)

Generated markdown
| Employee Name | Employee ID | Gross Pay Impact (NGN) |
|---------------|-------------|------------------------|
| ...           |             |                        |
| **Total New Hires** | | **(Total Value)** |
IGNORE_WHEN_COPYING_START
content_copy
download
Use code with caution.
Markdown
IGNORE_WHEN_COPYING_END
3.2 Variances for Continuing & Suspicious Employees

Create a table for all continuing employees. Highlight significant changes and flag suspicious records.

Generated markdown
| Employee Name | Employee ID | Gross Pay Variance (NGN) | Notes & Flags |
|---------------|-------------|--------------------------|---------------|
| ...           |             |                          | 🔴 **Suspicious (Identity Change):** Bank name changed from 'Old Bank' to 'New Bank'. |
| ...           |             |                          | 🔴 **Suspicious (Duplicate ID):** Employee ID appears X times in the current period. |
| ...           |             |                          | **Significant Pay Change:** Describe the change (e.g., Housing increased by NXX). |
| ...           |             |                          | No significant variance. |
IGNORE_WHEN_COPYING_START
content_copy
download
Use code with caution.
Markdown
IGNORE_WHEN_COPYING_END
4. Reconciliation of Gross Pay Variance

Summarize the drivers contributing to the total Gross Pay variance in a reconciliation table.

Generated markdown
| Driver                               | Count | Value Impact (NGN) |
|--------------------------------------|-------|--------------------|
| New Hires                            |       |                    |
| Departures                           |       |                    |
| Pay Changes (Continuing Employees)   |       |                    |
| Suspicious Anomalies (e.g., Duplicates) |       |                    |
| **Total Gross Pay Variance**         |       |                    |
IGNORE_WHEN_COPYING_START
content_copy
download
Use code with caution.
Markdown
IGNORE_WHEN_COPYING_END
5. Conclusion & Recommendations

Provide clear, numbered, and actionable recommendations based on your findings. Prioritize critical issues.

🔴 URGENT: Investigate Duplicate Employee ID: Detail the specific employee and the risk of double payment.

🔴 URGENT: Verify Bank Detail Change: Detail the specific employee and the potential fraud risk.

Review Pay Increase Authorization: Specify the employee and the amount that needs verification.

Data Cleansing Protocol: Recommend a future action to prevent similar data integrity issues.

Final Instructions:

Format all monetary values with the Nigerian currency symbol and two decimal places (e.g., N5,200.00).

Maintain a professional, objective, and data-driven tone. Your primary goal is to act as a diligent analyst, highlighting not just the numbers but the underlying data quality issues and operational risks they represent.
"""

    systemprompt1= retrieved_template6.format(old=old,new=new)

    responseY = llmv.invoke([
        systemprompt1,
        HumanMessage(content="Please review")
    ])
    return responseY.content



def get_payslips_from_json(json_file_path,desired_columns):
    # json_file_path is request.FILES.get('old') or request.FILES.get('new
    
    """
    Extracts payslips from a JSON file and returns a DataFrame with selected fields.
    
    Args:
        json_file_path (str): Path to the JSON file containing payroll data.
        
    Returns:
        pd.DataFrame: DataFrame containing the extracted payslips with selected fields.
    """
   
    data = json.load(json_file_path)
    if not isinstance(data, list):
     data = [data]
    #Notes
    all_payslips = []
    for payroll in data:
        payroll_id = payroll['_id']
        payslips = payroll.get('payslips', [])

        # Normalize payslips into a DataFrame
        df = pd.json_normalize(payslips)

        # Add payroll_id and payslip_id columns
        df['payroll_id'] = payroll_id
        df['employee_ID'] = df['_id']

        all_payslips.append(df)
    # Combine all into one DataFrame
    final_df = pd.concat(all_payslips, ignore_index=True)
    # desired_columns= desired_columns

    available_columns = [col for col in desired_columns if col in final_df.columns]
    # Trim safely using only available columns
    trimmed_df = final_df[available_columns]

    
    # Filter the final DataFrame
    # trimmed_df = final_df[desired_columns]
    json_output = trimmed_df.to_json(orient="records", indent=4)
    # # Save the final DataFrame to a CSV file
    # csv_file_path = r"C:\Users\Pro\Downloads\payslips_outputfullaboki.csv"
    # final_df.to_csv(csv_file_path, index=False)
        
    # df = pd.DataFrame(all_payslips)
    return json_output




        
# y = Prompt.objects.get(pk=1)  # Get the existing record
# retrieved_template1=y.response_prompt 
# response_prompt = retrieved_template1.format(
#     greeting=greeting,
#     ayula=state["messages"][-1].content,
#     attached_content=attached_content,  # Assuming no attached content for now
#     context=context,
#     pdf_text=pdf_text,
#     web_text=web_text,
#     query_answer=query_answer,

# )
# print("--- PROMPT ---",response_prompt)



# sys_msg = SystemMessage(content=response_prompt)

# model_with_structure = model.with_structured_output(Answer) 



# # Prepare key fields
# # key_fields = [line.strip() for line in importants.strip().splitlines()]
# key_fields=important()

# # Load raw JSON
# json_file_pathr = r"C:\Users\Pro\Desktop\Ayodele\25062025\myproject\myapp\raw.json"
# with open(json_file_pathr, 'r') as f:
#     datar = json.load(f)

# # Normalize JSON and extract desired fields
# payslips_dfr = pd.json_normalize(datar["payslips"])
# initial_json = payslips_dfr[key_fields].to_json(orient="records", indent=4)



# # Load raw JSON

# json_file_patht = r"C:\Users\Pro\Desktop\Ayodele\25062025\myproject\myapp\revised.json"
# with open(json_file_patht, 'r') as f:
#     datat = json.load(f)

# # Normalize JSON and extract desired fields
# payslips_dft =pd.json_normalize(datat["payslips"])
# treated_json = payslips_dft[key_fields].to_json(orient="records", indent=4)



# # Pass the JSON string to your function
# yemo = atb(initial_json,treated_json)
# print(yemo)

