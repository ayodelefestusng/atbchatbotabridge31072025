from django.conf import settings
from .models import Prompt,Prompt7
import os
import json
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_groq import ChatGroq

from dotenv import load_dotenv
load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# # Set environment variables for LangSmith
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GROQ_API_KEY=os.getenv("GROQ_API_KEY")
LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY")
LANGSMITH_PROJECT = os.getenv("LANGSMITH_PROJECT")
LANGSMITH_ENDPOINT = os.getenv("LANGSMITH_ENDPOINT")

os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_API_KEY"] = LANGSMITH_API_KEY if LANGSMITH_API_KEY else ""
os.environ["LANGSMITH_PROJECT"] = LANGSMITH_PROJECT if LANGSMITH_PROJECT else "Agent_Creation"
os.environ["LANGSMITH_ENDPOINT"] = LANGSMITH_ENDPOINT if LANGSMITH_ENDPOINT else "https://api.smith.langchain.com"
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY if GOOGLE_API_KEY else ""
os.environ["GROQ_API_KEY"] = GROQ_API_KEY if GROQ_API_KEY else ""


py = Prompt7.objects.get(pk=1)  # Get the existing record
google_model = py.views_model
llmvs = ChatGoogleGenerativeAI(model=google_model, temperature=0, google_api_key=GOOGLE_API_KEY)    
llmv=ChatGroq(model ="meta-llama/llama-4-scout-17b-16e-instruct", temperature=0,max_tokens=None,timeout=None,max_retries=2)



# py = Prompt.objects.get(pk=1)  # Get the existing record




# Django Core
from django.shortcuts import render, redirect, get_object_or_404
from django.http import (
    HttpResponse,
    JsonResponse,
    HttpResponseRedirect,
    HttpResponseServerError
)
from django.contrib.auth.decorators import login_required
from django.urls import reverse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_POST
from django.contrib import messages
from django.utils import timezone

# Django Auth (if used)
from django.contrib.auth import login, logout, authenticate
from django.contrib.auth.forms import PasswordResetForm
from django.contrib.auth.tokens import default_token_generator
from django.core.mail import send_mail


# Forms
from .forms import (
    RegistrationForm,
    PasswordSetupForm,
    PasswordChangeForm,
    PromptForm,
    ClientForm
)

# Models
from .models import (
    CustomUser,
    Conversation,
    Message,
    Insight,
    Ticket,
    Post,
    LLM,
    Prompt,
    Client
)

# NLP & AI Processing
from .chat_bot import process_message, safe_json,atb, important,get_payslips_from_json,desire

import pandas as pd

# WhatsApp / External Integration
import requests
from django.conf import settings

# NLP Libraries
import nltk
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
import spacy
import language_tool_python
from googletrans import Translator
from collections import defaultdict
from urllib.error import URLError

# Utilities
import os
import json
import logging
import uuid
import threading
# Optional: Unused but mentioned
# import pandas as pd
# import csv
# import openpyxl
# from django.core.paginator import Paginator


WHATSAPP_ACCESS_TOKEN = os.getenv('WHATSAPP_ACCESS_TOKEN')
WHATSAPP_PHONE_NUMBER_ID = os.getenv('WHATSAPP_PHONE_NUMBER_ID')
WHATSAPP_VERIFY_TOKEN = os.getenv('WHATSAPP_VERIFY_TOKEN')
WHATSAPP_API_VERSION = os.getenv('WHATSAPP_API_VERSION', 'v19.0')

import logging

logger = logging.getLogger(__name__)

def my_view(request):
    logger.info("This is an info message")
    logger.warning("This is a warning")
    logger.error("Something went wrong!")



import logging

# Get the logger instance for this module
logger = logging.getLogger(__name__)

def sample_view(request):
    logger.debug("Debugging info: Entered sample_view")
    logger.info("Info: Processing request")
    logger.warning("Warning: This is a sample warning message")
    logger.error("Error: Something went wrong")
    logger.critical("Critical: Major failure occurred!")

    return HttpResponse("Check your logs for messages!")



def register(request):
    if request.method == "POST":
        form = RegistrationForm(request.POST)
        if form.is_valid():
            user = form.save(commit=False)
            user.set_password(None)  # User sets password later
            user.save()

            token = default_token_generator.make_token(user)
            link = request.build_absolute_uri(reverse("setup_password", args=[user.pk, token]))

            send_mail(
                "Set Your Password",
                f"Click the link to set your password: {link}",
                "admin@example.com",
                [user.email],
            )

            return render(request, "myapp/registration_success.html", {"email": user.email})
    else:
        form = RegistrationForm()
    return render(request, "myapp/register.html", {"form": form})

def setup_password(request, user_id, token):
    user = CustomUser.objects.get(pk=user_id)
    if default_token_generator.check_token(user, token):
        if request.method == "POST":
            form = PasswordSetupForm(user, request.POST)
            if form.is_valid():
                form.save()
                return redirect("login")
        else:
            form = PasswordSetupForm(user)
        return render(request, "myapp/setup_password.html", {"form": form})
    else:
        return render(request, "myapp/error.html", {"message": "Invalid token"})

def password_reset_request(request):
    if request.method == "POST":
        form = PasswordResetForm(request.POST)
        if form.is_valid():
            email = form.cleaned_data["email"]
            user = CustomUser.objects.filter(email=email).first()
            if user:
                token = default_token_generator.make_token(user)
                link = request.build_absolute_uri(reverse("setup_password", args=[user.pk, token]))
                
                send_mail(
                    "Reset Your Password",
                    f"Click the link to reset your password: {link}",
                    "admin@example.com",
                    [email],
                )
            return render(request, "myapp/templates/password_reset_sent.html", {"email": email})
    else:
        form = PasswordResetForm()
    return render(request, "myapp/password_reset.html", {"form": form})

def change_password(request):
    if request.method == "POST":
        form = PasswordChangeForm(request.POST)
        if form.is_valid():
            user = authenticate(email=request.user.email, password=form.cleaned_data["old_password"])
            if user:
                user.set_password(form.cleaned_data["new_password"])
                user.save()
                logout(request)
                return redirect("login")
            else:
                return render(request, "myapp/change_password.html", {"form": form, "error": "Incorrect password"})
    else:
        form = PasswordChangeForm()
    return render(request, "myapp/change_password.html", {"form": form})



def user_login(request):
    if request.method == "POST":
        email = request.POST.get("email")
        password = request.POST.get("password")
        user = authenticate(request, email=email, password=password)

        if user:
            login(request, user)
            return redirect("home")  # Redirect to dashboard after login
        else:
            messages.error(request, "Invalid email or password. Please try again.")

    return render(request, "myapp/login.html")


@login_required
def user_logout(request):
    logout(request)
    messages.success(request, "You have been logged out successfully.")
    return redirect("login")  # Redirect to login page after logout



def about(request):
    return render(request,"myapp/about.html")

def contact(request):
    return render(request,"myapp/contact.html")


def home(request):
    return render(request,"myapp/home.html")





logger = logging.getLogger(__name__) # Initialize logger

# No custom_logout needed if entirely anonymous. If authentication remains for admin/other, keep it.
# def custom_logout(request):
#     logout(request)
#     return redirect('login')


def chat_home(request):
    # Ensure session key exists for anonymous users
    if not request.session.session_key:
        request.session.create()
    session_key = request.session.session_key

    # Get or create the active conversation for the current session
    conversation, _ = Conversation.objects.get_or_create(
        session_id=session_key,
        is_active=True,
        defaults={'is_active': True}
    )
    # Deactivate any other active conversations for this session (should ideally be unique anyway)
    Conversation.objects.filter(
        session_id=session_key, is_active=True
    ).exclude(id=conversation.id).update(is_active=False)

    messages = conversation.messages.all().order_by('timestamp')
    return render(request, 'chat.html', {'messages': messages, 'session_id': session_key}) # Pass session_id to template


def chat_home2(request):
    # Ensure session key exists for anonymous users
    if not request.session.session_key:
        request.session.create()
    session_key = request.session.session_key

    # Get or create the active conversation for the current session
    conversation, _ = Conversation.objects.get_or_create(
        session_id=session_key,
        is_active=True,
        defaults={'is_active': True}
    )
    # Deactivate any other active conversations for this session (should ideally be unique anyway)
    Conversation.objects.filter(
        session_id=session_key, is_active=True
    ).exclude(id=conversation.id).update(is_active=False)

    messages = conversation.messages.all().order_by('timestamp')
    return render(request, 'chat2.html', {'messages': messages, 'session_id': session_key}) # Pass session_id to template

@csrf_exempt
def send_message2(request):
    try:
        user_message = ''
        attachment = None

        # Ensure session key exists
        if not request.session.session_key:
            request.session.create()
        session_key = request.session.session_key

        # Handle JSON or form-data inputs
        if 'application/json' in request.content_type:
            try:
                if request.body:
                    data = json.loads(request.body)
                    user_message = data.get('message', '').strip()
            except json.JSONDecodeError:
                return JsonResponse({'status': 'error', 'response': 'Invalid JSON format'}, status=400)
        else:
            user_message = request.POST.get('message', '').strip()
            attachment = request.FILES.get('attachment')

        # Validate input
        if not user_message and not attachment:
            return JsonResponse({'status': 'error', 'response': 'Message or attachment is required'}, status=400)

        # Get or create conversation for the current session
        conversation, created = Conversation.objects.get_or_create(session_id=session_key, is_active=True, defaults={'is_active': True})

        # Deactivate other conversations if needed (unlikely with unique session_id)
        if not created:
            Conversation.objects.filter(session_id=session_key, is_active=True).exclude(id=conversation.id).update(is_active=False)

        # Save user message
        message = Message.objects.create(conversation=conversation, text=user_message, is_user=True)

        # Handle attachments
        file_path = "" # Default to empty string for process_message
        if attachment and hasattr(attachment, 'read'):
            message.attachment = attachment
            message.save()

            try:
                # Assuming you save attachments to MEDIA_ROOT
                # You might need to ensure the attachment is saved to disk before process_message accesses it
                file_path = message.attachment.path
            except Exception as e:
                logging.error(f"Error getting attachment path: {e}")
                file_path = "" # Reset file_path if there's an issue

        # Select correct message processing function
        try:
            # Pass session_key instead of request.userjso
            bot_response_data = process_message(user_message, session_key, file_path)

            bot_response_text = bot_response_data.get('messages', '')
            # metadata = bot_response_data.get('metadata', {}) # If you want to use metadata here

            if not bot_response_text:
                bot_response_text = "I'm sorry, I couldn't process your request."
            
        except Exception as e:
            bot_response_text = f"Error processing message: {str(e)}"
            logging.error(f"Message processing failed: {e}")

        # Save bot response
        Message.objects.create(conversation=conversation, text=bot_response_text, is_user=False)

        return JsonResponse({'status': 'success', 'response': bot_response_text, 'attachment_url': message.attachment.url if attachment else None})

    except Exception as e:
        logger.error(f"Server error in send_message: {e}", exc_info=True) # Log full traceback
        return JsonResponse({'status': 'error', 'response': f"Server error: {str(e)}"}, status=500)


def summary(request):
    # Ensure session key exists
    if not request.session.session_key:
        request.session.create()
    session_key = request.session.session_key
    # Fetch all insights for the current session
    insights = Insight.objects.filter(session_id=session_key).order_by('-updated_at')
    return render(request, 'analytics.html', {'insight': insights})


def send_message3(request, param_name, session_key): # Changed session to session_key
    """Sends a message with a default session_key."""
    
    file_path = ""   # Default file path (empty)
    user_message = param_name
 
    # Use the provided session_key directly
    # bot_response is expected to be a dict from nlp_processor
    bot_response = process_message(user_message, session_key, file_path)
    bot_response_metadata = bot_response.get('metadata', {})

    return JsonResponse({'status': 'success', 'response': bot_response_metadata}) # Return metadata

@csrf_exempt
def send_messageUsedForImage(request):
    try:
        user_message, attachment = '', None

        # 📍 Ensure session key is created
        if not request.session.session_key:
            request.session.create()
        session_key = request.session.session_key

        # 📨 Handle input from JSON or form-data
        if 'application/json' in request.content_type:
            try:
                if request.body:
                    data = json.loads(request.body)
                    user_message = data.get('message', '').strip()
            except json.JSONDecodeError:
                return JsonResponse({'status': 'error', 'response': 'Invalid JSON format'}, status=400)
        else:
            user_message = request.POST.get('message', '').strip()
            attachment = request.FILES.get('attachment')

        # 🚫 Validate input
        if not user_message and not attachment:
            return JsonResponse({'status': 'error', 'response': 'Message or attachment is required'}, status=400)

        # 🗃️ Create initial user message object (used for attachment only)
        conversation, _ = Conversation.objects.get_or_create(session_id=session_key, is_active=True)
        user_msg_obj = Message.objects.create(
            conversation=conversation,
            text=user_message,
            is_user=True
        )

        # 📎 Handle attachment upload (if any)
        file_path = ""
        if attachment and hasattr(attachment, 'read'):
            user_msg_obj.attachment = attachment
            user_msg_obj.save()
            try:
                file_path = user_msg_obj.attachment.path
            except Exception as e:
                logging.warning(f"Could not resolve attachment path: {e}")
                file_path = ""

        # 🧠 Process message with LLM
        
        try:
            bot_response_data = process_message(user_message, session_key, file_path)
            bot_text = bot_response_data.get('messages', '')
            if not bot_text:
                bot_text = "I'm sorry, I couldn't process your request."
        except Exception as e:
            bot_text = f"Error processing message: {str(e)}"
            logging.error(f"process_message failed: {e}")

        # ✅ Return response immediately
        response_payload = {
            'status': 'success',
            'response': bot_text,
            'attachment_url': user_msg_obj.attachment.url if attachment else None
        }
        JsonResponseReady = JsonResponse(response_payload)
       
        # Log the response for debugging
        logging.info(f"Bot response: {bot_text}")
        
        # Return the response immediately
        # return JsonResponseReady

          # <-- Ensure this is always defined

        # 🧵 Background thread to save bot message + insights
        def save_async(bot_response_data):
            try:
                 # Bot reply message
                Message.objects.create(
                    conversation=conversation,
                    text=bot_text,
                    is_user=False
                )
                

                # Metadata insight (if present)
                metadata = bot_response_data.get('metadata', {})
                if metadata:
                    insight_obj = Insight.objects.filter(session_id=session_key).order_by('-updated_at').first()
                    if not insight_obj:
                        insight_obj = Insight(session_id=session_key)

                    # Populate fields
                    insight_obj.question      = metadata.get('question', '')
                    insight_obj.answer        = metadata.get('answer', '')
                    insight_obj.sentiment     = metadata.get('sentiment',0)
                    insight_obj.ticket        = safe_json(metadata.get('ticket', {}))
                    insight_obj.source        = safe_json(metadata.get('source', {}))
                    insight_obj.summary       = metadata.get('summary', '')
                    insight_obj.sum_sentiment = safe_json(metadata.get('sum_sentiment', 0))
                    insight_obj.sum_ticket    = safe_json(metadata.get('sum_ticket', {}))
                    insight_obj.sum_source    = safe_json(metadata.get('sum_source', {}))

                    insight_obj.save()
            except Exception as e:
                logging.error(f"Background DB save failed: {e}", exc_info=True)

        threading.Thread(target=save_async, args=(bot_response_data,)).start()  # 🚀 Launch async task

        return JsonResponseReady

    except Exception as e:
        logging.error(f"Fatal server error: {e}", exc_info=True)
        return JsonResponse({'status': 'error', 'response': f"Server error: {str(e)}"}, status=500)


# from django.views.decorators.csrf import csrf_exempt # Uncomment if you need csrf_exempt, but user requested not to use it.

# Assuming these models and functions are defined elsewhere in your Django project:
# from .models import Conversation, Message, Insight
# from .utils import safe_json, process_message

# Configure logging (it's good practice to have this configured globally in settings.py or app.py)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# @csrf_exempt # User requested not to use async or csrf_exempt
@csrf_exempt
def send_message(request):
    """
    Handles incoming user messages, processes them with an LLM,
    sends an immediate response to the user, and then saves
    the bot's message and any associated metadata in a background thread.
    """
    try:
        user_message, attachment = '', None

        # 📍 Ensure session key is created for conversation tracking
        if not request.session.session_key:
            request.session.create()
        session_key = request.session.session_key

        # 📨 Handle input from JSON payload or form-data
        if 'application/json' in request.content_type:
            try:
                if request.body:
                    data = json.loads(request.body)
                    user_message = data.get('message', '').strip()
            except json.JSONDecodeError:
                # Return error if JSON is malformed
                return JsonResponse({'status': 'error', 'response': 'Invalid JSON format'}, status=400)
        else:
            # Handle form-data (e.g., from a web form submission)
            user_message = request.POST.get('message', '').strip()
            attachment = request.FILES.get('attachment')

        # 🚫 Validate input: either a message or an attachment must be present
        if not user_message and not attachment:
            return JsonResponse({'status': 'error', 'response': 'Message or attachment is required'}, status=400)

        # 🗃️ Get or create the conversation for the current session
        conversation, _ = Conversation.objects.get_or_create(session_id=session_key, is_active=True)

        # Create and save the user's message immediately
        user_msg_obj = Message.objects.create(
            conversation=conversation,
            text=user_message,
            is_user=True
        )

        # 📎 Handle attachment upload (if any)
        file_path = ""
        if attachment and hasattr(attachment, 'read'):
            user_msg_obj.attachment = attachment
            user_msg_obj.save() # Save the attachment to the user message object
            try:
                file_path = user_msg_obj.attachment.path
            except Exception as e:
                # Log a warning if the attachment path cannot be resolved
                logging.warning(f"Could not resolve attachment path: {e}")
                file_path = ""

        # 🧠 Process the user's message with the Language Model (LLM)
        bot_response_data = {}  # <-- Ensure this is always defined

        try:
            # Check if user is asking for a chart
            if any(word in user_message.lower() for word in ['chart', 'graph', 'plot', 'visualize']):
                # Use the chart generation function
                from .kip import generate_chart
                chart_text, chart_image = generate_chart(user_message)
                
                # Create HTML response with image
                if chart_image:
                    html_response = f"""
                    <div class="chart-response">
                        <div class="analysis-text">{chart_text}</div>
                        <div class="chart-image">
                            <img src="data:image/png;base64,{chart_image}" 
                                 alt="Generated Chart" 
                                 class="img-fluid" 
                                 style="max-width: 100%; height: auto;">
                        </div>
                    </div>
                    """
                    bot_text = html_response
                else:
                    bot_text = chart_text
            else:
                # Use regular message processing
                bot_response_data = process_message(user_message, session_key, file_path)
                bot_text = bot_response_data.get('messages', '')
                
            if not bot_text:
                bot_text = "I'm sorry, I couldn't process your request."
        except Exception as e:
            # Handle errors during LLM processing
            bot_text = f"Error processing message: {str(e)}"
            logging.error(f"process_message failed: {e}", exc_info=True)

        # ✅ Create and save the bot's response message immediately for efficiency
        # This ensures the bot's reply is recorded quickly, separate from metadata.
        Message.objects.create(
            conversation=conversation,
            text=bot_text,
            is_user=False
        )

        # ✅ Prepare the JSON response payload to send back to the user
        response_payload = {
            'status': 'success',
            'response': bot_text,
            'attachment_url': user_msg_obj.attachment.url if attachment else None
        }
        JsonResponseReady = JsonResponse(response_payload)

        # Log the bot's response for debugging purposes
        logging.info(f"Bot response: {bot_text}")

        # 🧵 Define a function for background saving of metadata
        def save_metadata_async(bot_response_data_for_thread, current_session_key):
            """
            Saves bot response metadata (insights) in a separate thread.
            This prevents blocking the main request-response cycle.
            """
            try:
                metadata = bot_response_data_for_thread.get('metadata', {})
                if metadata:
                    # Retrieve the latest Insight object for the session or create a new one
                    insight_obj = Insight.objects.filter(session_id=current_session_key).order_by('-updated_at').first()
                    if not insight_obj:
                        insight_obj = Insight(session_id=current_session_key)

                    # Populate Insight object fields from metadata
                    insight_obj.question      = metadata.get('question', '')
                    insight_obj.answer        = metadata.get('answer', '')
                    insight_obj.sentiment     = metadata.get('sentiment', 0)
                    insight_obj.ticket        = safe_json(metadata.get('ticket', {}))
                    insight_obj.source        = safe_json(metadata.get('source', {}))
                    insight_obj.summary       = metadata.get('summary', '')
                    insight_obj.sum_sentiment = safe_json(metadata.get('sum_sentiment', 0))
                    insight_obj.sum_ticket    = safe_json(metadata.get('sum_ticket', {}))
                    insight_obj.sum_source    = safe_json(metadata.get('sum_source', {}))

                    insight_obj.save() # Save the updated or new Insight object
            except Exception as e:
                # Log any errors encountered during background saving
                logging.error(f"Background DB save failed: {e}", exc_info=True)

        # 🚀 Launch the background thread to save metadata
        # Pass bot_response_data and session_key as arguments to the thread function
        threading.Thread(target=save_metadata_async, args=(bot_response_data, session_key,)).start()

        # Return the JSON response to the user immediately
        return JsonResponseReady

    except Exception as e:
        # Catch any fatal server errors and return a 500 response
        logging.error(f"Fatal server error: {e}", exc_info=True)
        return JsonResponse({'status': 'error', 'response': f"Server error: {str(e)}"}, status=500)






def chat_history(request):
    # Ensure session key exists
    if not request.session.session_key:
        request.session.create()
    session_key = request.session.session_key

    # Fetch conversations for the current session
    conversations = Conversation.objects.filter(session_id=session_key).order_by('-updated_at')
    return render(request, 'history.html', {'conversations': conversations})

def view_conversation(request, conversation_id):
    # Ensure session key exists
    if not request.session.session_key:
        request.session.create()    
    session_key = request.session.session_key

    # Fetch conversation for the current session and ID
    conversation = get_object_or_404(Conversation, id=conversation_id, session_id=session_key)
    messages = conversation.messages.all().order_by('timestamp')
    return render(request, 'chatbot/conversation.html', {
        'messages': messages,
        'conversation': conversation
    })


def oya(request):
    # Ensure session key exists
    if not request.session.session_key:
        request.session.create()
    session_key = request.session.session_key

    # Fetch the latest Insight for the current session_id
    # Note: If Insight.session_id is not unique, `.latest()` might not be suitable
    # if you expect multiple insights per session to be distinct.
    # If it's "the latest insight *ever* generated for this session", then .latest() works.
    
    # Using .filter() and .first() to get the single latest insight
    insight_latest = Insight.objects.filter(session_id=session_key).order_by('-updated_at').first()
    
    context = {
        'insight': insight_latest, # Now a single object or None
        'yemi': "insight_data", # Renamed for clarity if 'yemi' is generic
    }
    return render(request, 'analytics.html', context)


@csrf_exempt
def send_message111(request):
    try:
        user_message = ''
        attachment = None

        # Ensure session key exists
        if not request.session.session_key:
            request.session.create()
        session_key = request.session.session_key

        # Handle JSON or form-data inputs
        if 'application/json' in request.content_type:
            try:
                if request.body:
                    data = json.loads(request.body)
                    user_message = data.get('message', '').strip()
            except json.JSONDecodeError:
                return JsonResponse({'status': 'error', 'response': 'Invalid JSON format'}, status=400)
        else:
            user_message = request.POST.get('message', '').strip()
            attachment = request.FILES.get('attachment')

        # Validate input
        if not user_message and not attachment:
            return JsonResponse({'status': 'error', 'response': 'Message or attachment is required'}, status=400)

        # Get or create conversation for the current session
        conversation, created = Conversation.objects.get_or_create(session_id=session_key, is_active=True, defaults={'is_active': True})

        # Deactivate other conversations if needed (unlikely with unique session_id)
        if not created:
            Conversation.objects.filter(session_id=session_key, is_active=True).exclude(id=conversation.id).update(is_active=False)

        # Save user message
        message = Message.objects.create(conversation=conversation, text=user_message, is_user=True)

        # Handle attachments
        file_path = "" # Default to empty string for process_message
        if attachment and hasattr(attachment, 'read'):
            message.attachment = attachment
            message.save()

            try:
                # Assuming you save attachments to MEDIA_ROOT
                # You might need to ensure the attachment is saved to disk before process_message accesses it
                file_path = message.attachment.path
            except Exception as e:
                logging.error(f"Error getting attachment path: {e}")
                file_path = "" # Reset file_path if there's an issue

        # Select correct message processing function
        try:
            # Pass session_key instead of request.user
            bot_response_data = process_message(user_message, session_key, file_path)

            bot_response_text = bot_response_data.get('messages', '')
            

            if not bot_response_text:
                bot_response_text = "I'm sorry, I couldn't process your request."
            
        except Exception as e:
            bot_response_text = f"Error processing message: {str(e)}"
            logging.error(f"Message processing failed: {e}")

        # Save bot response
        Message.objects.create(conversation=conversation, text=bot_response_text, is_user=False)

        return JsonResponse({'status': 'success', 'response': bot_response_text, 'attachment_url': message.attachment.url if attachment else None})

    except Exception as e:
        logger.error(f"Server error in send_message: {e}", exc_info=True) # Log full traceback
        return JsonResponse({'status': 'error', 'response': f"Server error: {str(e)}"}, status=500)



def post_list(request):
    # This view doesn't directly use user/session for filtering, so it remains largely same.
    # If posts need to be anonymous-session-specific, add session_id filtering here too.
    posts = "Post.objects.all()" # This is a string, not fetching actual Post objects
    # You likely want: posts = Post.objects.all()
    return render(request, 'post_list.html', {'posts': posts})


def homey(request):
    return render(request, "home.html")


def edit_currency(request):
    category_id = request.POST.get('currency')
    cont = LLM.objects.get(pk=1)
    cont.currency = category_id
    cont.save()
    # Corrected redirect (assuming a URL named 'chat_home')
    return redirect(reverse('chat_home')) # Or redirect('/chat/') if you have a fixed URL path


# Removed @login_required decorators if dashboard is also anonymous
def dashboard(request):
    # If dashboard is generic, no session_key needed. If session-specific, add it.
    return render(request, 'dashboard.html')

def prompt_list(request):
    prompts = Prompt.objects.all()
    return render(request, 'prompt_list.html', {'prompts': prompts})

def update_prompt(request, prompt_id):
    prompt = get_object_or_404(Prompt, id=prompt_id)
    if request.method == "POST":
        form = PromptForm(request.POST, instance=prompt)
        if form.is_valid():
            form.save()
            return redirect('prompt_list')  # Redirect after update
    else:
        form = PromptForm(instance=prompt)
    return render(request, 'update_prompt.html', {'form': form, 'prompt': prompt})


def client_list(request):
    clients = Client.objects.all()
    return render(request, "client_list.html", {"clients": clients})

def create_client(request):
    if request.method == "POST":
        form = ClientForm(request.POST, request.FILES)
        if form.is_valid():
            form.save()
            return redirect("client_list")
    else:
        form = ClientForm()
    return render(request, "create_client.html", {"form": form})

def update_client(request, client_id):
    client = get_object_or_404(Client, id=client_id)
    if request.method == "POST":
        form = ClientForm(request.POST, request.FILES, instance=client)
        if form.is_valid():
            form.save()
            return redirect("client_list")
    else:
        form = ClientForm(instance=client)
    return render(request, "update_client.html", {"form": form, "client": client})


def health_check(request):
    return HttpResponse("OK")

def gemini_view(request):
    try:
        response_text = "Gemini API response goes here"
        return HttpResponse(response_text)
    except Exception as e:
        logger.exception("Gemini API call failed")
        return HttpResponseServerError("Something went wrong")

# Add new view for chart display
def chart_view(request):
    """View to display charts generated from kip.py"""
    try:
        # Import your kip.py function (you'll need to modify kip.py to return the data)
        from .kip import generate_chart  # You'll need to create this function
        
        # Get chart data
        chart_data = generate_chart()  # This should return (text, image_base64)
        
        context = {
            'chart_text': chart_data[0] if chart_data else '',
            'chart_image': chart_data[1] if chart_data and len(chart_data) > 1 else None,
        }
        
        return render(request, 'myapp/chart_display.html', context)
        
    except Exception as e:
        logger.exception("Chart generation failed")
        return render(request, 'myapp/chart_display.html', {
            'error': str(e),
            'chart_text': '',
            'chart_image': None
        })


# Initialize logger for this module (as defined in your settings.py)
logger = logging.getLogger(__name__)

# Load WhatsApp API settings from environment variables
# These should be loaded by dotenv in settings.py and then available via os.getenv()



# Initialize logger for this module

# Load WhatsApp API settings from environment variables


@csrf_exempt
def whatsapp_webhook(request):
    """
    Handles WhatsApp Cloud API webhook requests.
    - GET: For webhook verification.
    - POST: For incoming messages.
    This version sends a quick, fixed reply to avoid timeouts.
    """

    if request.method == 'GET':
        mode = request.GET.get('hub.mode')
        token = request.GET.get('hub.verify_token')
        challenge = request.GET.get('hub.challenge')

        if mode == 'subscribe' and token == WHATSAPP_VERIFY_TOKEN:
            logger.info("WhatsApp webhook verification successful.")
            return HttpResponse(challenge, status=200)
        else:
            logger.warning(f"WhatsApp webhook verification failed. Mode: {mode}, Token: {token}")
            return HttpResponse("Failed verification", status=403)

    elif request.method == 'POST':
        try:
            data = json.loads(request.body)
            logger.info(f"Received WhatsApp webhook data: {json.dumps(data, indent=2)}")

            if 'object' in data and 'entry' in data:
                for entry in data['entry']:
                    for change in entry.get('changes', []):
                        if 'value' in change and 'messages' in change['value']:
                            for msg_data in change['value']['messages']:
                                sender_wa_id = msg_data['from']
                                message_id = msg_data['id'] # Get message ID for read receipt

                                # --- Immediately mark message as read ---
                                mark_whatsapp_message_as_read(message_id)

                                # --- Send a quick, fixed reply ---
                                fixed_reply_text = "Hello from your quick bot! I received your message. (This is a test reply to confirm delivery)."
                                logger.info(f"Attempting to send fixed reply to {sender_wa_id}.")
                                send_whatsapp_message(sender_wa_id, fixed_reply_text)

                                # You can add a print statement here for immediate confirmation in terminal
                                print(f"--- Sent quick reply to {sender_wa_id}: {fixed_reply_text} ---")

                                # No NLP processing, no media handling in this basic version
                                # We exit the loop early after the first message for simplicity
                                break
                            break
                        # Handle status updates (e.g., delivered, read receipts for messages sent by bot)
                        elif 'value' in change and 'statuses' in change['value']:
                            status_data = change['value']['statuses'][0] # Take the first status for logging
                            logger.info(f"Received WhatsApp status update: ID={status_data['id']}, Status={status_data['status']}, Recipient={status_data.get('recipient_id')}")
                            # You could process these further if needed, e.g., update your DB about message delivery status
                    break # Break from outer loop after first entry for simplicity
            
            # Always return a 200 OK immediately after processing
            return JsonResponse({'status': 'success', 'message': 'Webhook processed quickly'}, status=200)

        except json.JSONDecodeError:
            logger.error("Invalid JSON received in WhatsApp webhook POST request.")
            return HttpResponse("Invalid JSON", status=400)
        except Exception as e:
            logger.exception(f"Unexpected error in WhatsApp webhook POST request: {e}")
            return HttpResponse("Internal Server Error", status=500)
    else:
        logger.warning(f"Received unsupported HTTP method: {request.method}")
        return HttpResponse("Method Not Allowed", status=405)


def download_whatsapp_media(media_id: str, sender_wa_id: str):
    """
    Dummy function for this basic test. Will not be called.
    """
    logger.warning("download_whatsapp_media called in basic test version. This should not happen.")
    return None

def send_whatsapp_message(recipient_wa_id: str, message_text: str):
    """
    Sends a text message back to the WhatsApp user via Cloud API.
    """
    url = f"https://graph.facebook.com/{WHATSAPP_API_VERSION}/{WHATSAPP_PHONE_NUMBER_ID}/messages"
    headers = {
        "Authorization": f"Bearer {WHATSAPP_ACCESS_TOKEN}",
        "Content-Type": "application/json",
    }
    payload = {
        "messaging_product": "whatsapp",
        "to": recipient_wa_id,
        "type": "text",
        "text": {"body": message_text},
    }

    try:
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()
        logger.info(f"Message sent to {recipient_wa_id}. Response: {response.json()}")
        return True
    except requests.exceptions.RequestException as e:
        logger.error(f"Error sending WhatsApp message to {recipient_wa_id}: {e}")
        if response is not None:
            logger.error(f"WhatsApp API Error Response: {response.text}")
        return False
    except Exception as e:
        logger.exception(f"Unexpected error in send_whatsapp_message for {recipient_wa_id}: {e}")
        return False

def mark_whatsapp_message_as_read(message_id: str):
    """
    Marks a specific WhatsApp message as read via Cloud API.
    """
    url = f"https://graph.facebook.com/{WHATSAPP_API_VERSION}/{WHATSAPP_PHONE_NUMBER_ID}/messages"
    headers = {
        "Authorization": f"Bearer {WHATSAPP_ACCESS_TOKEN}",
        "Content-Type": "application/json",
    }
    payload = {
        "messaging_product": "whatsapp",
        "status": "read",
        "message_id": message_id
    }
    try:
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()
        logger.info(f"Message {message_id} marked as read.")
        return True
    except requests.exceptions.RequestException as e:
        logger.error(f"Error marking message {message_id} as read: {e}")
        if response is not None:
            logger.error(f"WhatsApp API Read Receipt Error Response: {response.text}")
        return False
    


# --- NLP/NLTK/Translation Initialization (AJADI) ---


# Use googletrans for translation (unofficial, unstable)

# Set up NLTK data path (optional, for deployment)
nltk_static_path = os.path.join(os.path.dirname(__file__), "static", "nltk")
if nltk_static_path not in nltk.data.path:
    nltk.data.path.append(nltk_static_path)

# Check required NLTK packages
required_nltk = [
    'corpora/wordnet',
    'corpora/omw-1.4',
    'tokenizers/punkt',
    'taggers/averaged_perceptron_tagger'
]
for pkg in required_nltk:
    try:
        nltk.data.find(pkg)
    except LookupError:
        print(f"⚠️ NLTK package '{pkg}' not found. Please run 'nltk.download()' to install it.")

# Initialize spaCy
try:
    nlp = spacy.load("en_core_web_sm")
except Exception as e:
    nlp = None
    print(f"spaCy model load error: {e}")

# Initialize LanguageTool
try:
    tool = language_tool_python.LanguageTool('en-US')
except Exception as e:
    tool = None
    print(f"LanguageTool init error: {e}")

lemmatizer = WordNetLemmatizer()

# Initialize googletrans Translator
try:
    translator = Translator()
except Exception as e:
    translator = None
    print(f"googletrans Translator init error: {e}")



# --- NLTK Data Downloads (Run once when the server starts) ---
# print("--- Starting data downloads ---")
# try:
#     print("Attempting to download NLTK data (wordnet, omw-1.4, punkt, averaged_perceptron_tagger)...")
#     nltk.download('wordnet', quiet=True)
#     nltk.download('omw-1.4', quiet=True)
#     nltk.download('punkt', quiet=True)
#     nltk.download('averaged_perceptron_tagger', quiet=True)
#     print("NLTK data downloaded successfully or already present.")
# except URLError as e:
#     print(f"NLTK Download Error (URLError): Could not reach download server. Check your internet connection. Error: {e}")
# except Exception as e:
#     print(f"An unexpected error occurred during NLTK download: {e}")
# print("--- Data downloads complete ---")


print("--- Checking NLTK data availability ---")

# Optional: Explicitly set the path to your NLTK data folder

nltk_static_path = os.path.join(os.path.dirname(__file__), "static", "nltk")
nltk.data.path.append(nltk_static_path)

# Optional: verify packages
required = ['corpora/wordnet', 'corpora/omw-1.4', 'tokenizers/punkt', 'taggers/averaged_perceptron_tagger']

for package in required:
    try:
        nltk.data.find(package)
        print(f"✔️ NLTK package '{package}' found.")
    except LookupError:
        print(f"⚠️ NLTK package '{package}' not found. Please run 'nltk.download()' to install it.")

# ✅ Check if required corpora are present
required_packages = [
    'corpora/wordnet',
    'corpora/omw-1.4',
    'tokenizers/punkt',
    'taggers/averaged_perceptron_tagger'
]

missing = []

for package in required_packages:
    try:
        nltk.data.find(package)
    except LookupError:
        missing.append(package)

if missing:
    print(f"⚠️ Missing NLTK packages: {missing}")
    print("Consider running a setup script to download them once.")
else:
    print("✔️ All required NLTK data found locally.")

# Initialize NLP tools
nlp = None
tool = None
try:
    nlp = spacy.load("en_core_web_sm")
    print("spaCy model 'en_core_web_sm' loaded successfully.")
except Exception as e:
    print(f"Error loading spaCy model 'en_core_web_sm'. Please ensure it's installed ('python -m spacy download en_core_web_sm'). Features relying on spaCy might not work. Error: {e}")

try:
    tool = language_tool_python.LanguageTool('en-US')
    print("LanguageTool initialized successfully.")
except Exception as e:
    print(f"Error initializing LanguageTool. Please ensure it's accessible and Java is installed on your system. Features relying on LanguageTool might not work. Error: {e}")

lemmatizer = WordNetLemmatizer()

# Initialize googletrans Translator
translator = None
try:
    translator = Translator()
    print("googletrans Translator initialized.")
except Exception as e:
    print(f"Error initializing googletrans Translator: {e}")
    print("Translation functionality might be unstable or fail.")


def editor_view(request):
    """
    Renders the main HTML editor page.
    """
    return render(request, 'myapp/editor.html')


def _get_text_analysis_results(text):
    """
    Performs all text analysis (grammar, suggestions, completions, thesaurus)
    and returns a dictionary of results.
    """
    results = {
        'highlighted_spans': [],
        'suggestions': [],
        'corrections': {},
        'sentence_completions': [],
        'grammar_checks': [],
        'thesaurus_suggestions': {},
    }

    if not text.strip():
        return results

    doc = None
    if nlp:
        try:
            doc = nlp(text)
        except Exception as e:
            print(f"Error processing text with spaCy: {e}")
            results['suggestions'].append("SpaCy error during processing.")
            results['thesaurus_suggestions'] = {"_note": "SpaCy error during processing."}
    else:
        results['suggestions'].append("SpaCy not available for sentence suggestions.")
        results['thesaurus_suggestions'] = {"_note": "SpaCy not available for thesaurus suggestions."}


    matches = []
    if tool:
        try:
            matches = tool.check(text)
        except Exception as e:
            print(f"Error checking text with LanguageTool: {e}")
            results['grammar_checks'].append({'message': 'LanguageTool error during check.'})
    else:
        print("LanguageTool not initialized, skipping grammar and spelling checks.")


    # --- Feature 1: Grammar Checks & Highlighting (from LanguageTool) ---
    highlight_spans = []
    corrections = {}
    grammar_checks = []
    for match in matches:
        if hasattr(match, 'ruleId') and match.ruleId == 'MORFOLOGIK_RULE_EN_US':
            highlight_spans.append({
                'start': match.offset,
                'end': match.offset + match.errorLength,
                'message': match.message
            })

        if match.replacements:
            error_word = text[match.offset:match.offset + match.errorLength]
            if error_word not in corrections:
                corrections[error_word] = match.replacements[0]

        context_start = match.context.offset if hasattr(match.context, 'offset') else match.offset
        context_length = match.context.length if hasattr(match.context, 'length') else match.errorLength
        grammar_checks.append({
            'message': match.message,
            'context': text[context_start : context_start + context_length],
            'replacements': match.replacements
        })
    results['highlighted_spans'] = highlight_spans
    results['corrections'] = corrections
    results['grammar_checks'] = grammar_checks

    # --- Feature 2: Sentence Suggestions (simple rephrasing using spaCy/WordNet) ---
    sentence_suggestions = []
    if doc:
        for sent in doc.sents:
            original_sentence = sent.text
            rephrased_sentence = original_sentence
            for token in sent:
                if token.pos_ == "NOUN":
                    synonyms = []
                    try:
                        for syn in wordnet.synsets(token.text):
                            for lemma in syn.lemmas():
                                if lemma.name().lower() != token.text.lower():
                                    synonyms.append(lemma.name().replace('_', ' '))
                    except Exception as e:
                        print(f"Error getting synonyms for '{token.text}': {e}")
                    if synonyms and token.text.lower() != synonyms[0].lower():
                        rephrased_sentence = rephrased_sentence.replace(token.text, synonyms[0], 1)
                        break
            if rephrased_sentence != original_sentence:
                sentence_suggestions.append(rephrased_sentence)
    results['suggestions'] = sentence_suggestions


    # --- Feature 3: Sentence Completions (basic N-gram like prediction) ---
    words = text.split()
    sentence_completions = []
    if len(words) > 0:
        last_word = words[-1].lower()
        if last_word.endswith('.'):
            sentence_completions.extend(["The", "It", "However,"])
        elif last_word == "hello":
            sentence_completions.append("world")
        elif last_word == "how":
            sentence_completions.append("are you")
        elif last_word == "i":
            sentence_completions.extend(["am", "want to"])
        elif last_word == "the":
            sentence_completions.extend(["cat", "dog", "house"])
    results['sentence_completions'] = sentence_completions


    # --- Feature 4: Thesaurus Suggestions (Synonyms/Antonyms using WordNet) ---
    thesaurus_suggestions = defaultdict(list)
    if doc:
        for token in doc:
            if token.pos_ in ["NOUN", "VERB", "ADJ", "ADV"]:
                synonyms = set()
                antonyms = set()
                try:
                    for syn in wordnet.synsets(token.text):
                        for lemma in syn.lemmas():
                            if lemma.name().lower() != token.text.lower():
                                synonyms.add(lemma.name().replace('_', ' '))
                            for antonym_lemma in lemma.antonyms():
                                antonyms.add(antonym_lemma.name().replace('_', ' '))
                except Exception as e:
                    print(f"Error finding synonyms/antonyms for '{token.text}': {e}")

                if synonyms:
                    thesaurus_suggestions[token.text].extend(sorted(list(synonyms)))
                if antonyms:
                    thesaurus_suggestions[token.text].extend(sorted(list(antonyms)))
        results['thesaurus_suggestions'] = dict(thesaurus_suggestions)
    else:
        results['thesaurus_suggestions'] = {"_note": "SpaCy not available for thesaurus suggestions."}

    return results


def process_text(request):
    """
    Processes user input text for grammar, suggestions, completions, and thesaurus.
    Returns JSON for POST requests (AJAX) or HTML for GET requests (direct browser).
    """
    if request.method == "POST":
        try:
            data = json.loads(request.body)
            text = data.get('text', '')
        except json.JSONDecodeError:
            return JsonResponse({'error': 'Invalid JSON in request body'}, status=400)

        results = _get_text_analysis_results(text)
        return JsonResponse(results)

    elif request.method == "GET":
        text = request.GET.get('text', '') # Get text from URL query parameter
        results = _get_text_analysis_results(text)
        return render(request, 'myApp/results.html', {'text_input': text, 'results': results})
    else:
        return JsonResponse({'error': 'Method not allowed'}, status=405)


import json
import asyncio
import logging

from django.http import JsonResponse
from googletrans import Translator # Assuming googletrans is installed

# Initialize the translator globally.
# For googletrans, initializing it once is generally fine.
translator = Translator()

# Configure logging (good practice for debugging)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

import json
import asyncio
import threading
import logging
from queue import Queue # Used to pass results back from the thread

from django.http import JsonResponse
from googletrans import Translator # Assuming googletrans is installed

# Initialize the translator globally.
# For googletrans, initializing it once is generally fine.
translator = Translator()

# Configure logging (good practice for debugging)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def _translate_in_thread(text, lang, result_queue):
    """
    Helper function to execute the asynchronous translation
    within a separate, dedicated thread.
    This ensures that asyncio.run() operates in an isolated event loop,
    preventing "Event loop is closed" errors in a multi-threaded environment.
    """
    try:
        # asyncio.run() creates and manages its own event loop for the coroutine.
        # This is the key to running an async function from a synchronous context
        # without interfering with the main thread's event loop (if any).
        translated = asyncio.run(translator.translate(text, dest=lang))
        # Put the successful result into the queue to be retrieved by the main thread
        result_queue.put({'success': True, 'translated_text': translated.text})
    except Exception as e:
        # Log any errors that occur within the translation thread
        logging.exception(f"Error during googletrans API call in background thread: {e}")
        # Put an error message into the queue if translation fails
        result_queue.put({'success': False, 'error': f"Translation failed: {e}. The 'googletrans' library is known to be unstable. If this error persists, consider using official translation APIs for reliability."})

def translate_text(request):
    """
    Translates the given text into a target language using the googletrans library.
    This function is designed to be synchronous from Django's perspective,
    offloading the asynchronous translation logic to a separate thread.
    """
    if request.method == "POST":
        try:
            data = json.loads(request.body)
            text = data.get('text', '').strip()
            lang = data.get('lang', 'fr').strip()

            # Validate input: ensure text is provided
            if not text:
                return JsonResponse({'error': 'Text to translate is required'}, status=400)

            # Create a Queue to receive the result from the background thread
            result_queue = Queue()

            # Create and start a new thread to perform the translation
            translation_thread = threading.Thread(
                target=_translate_in_thread,
                args=(text, lang, result_queue)
            )
            translation_thread.start()

            # Wait for the thread to complete, with a timeout to prevent indefinite hanging
            # A reasonable timeout (e.g., 30 seconds) is crucial for responsiveness.
            translation_thread.join(timeout=30)

            # Check if the thread is still alive after the timeout
            if translation_thread.is_alive():
                logging.error("Translation thread timed out.")
                return JsonResponse({'error': 'Translation service timed out. Please try again.'}, status=504)

            # Retrieve the result from the queue. This will block until an item is available.
            result = result_queue.get()

            # Check the success status from the thread's result
            if result['success']:
                return JsonResponse({'translated_text': result['translated_text']})
            else:
                # Return the error message provided by the background thread
                return JsonResponse({'error': result['error']}, status=500)

        except json.JSONDecodeError:
            # Handle cases where the request body is not valid JSON
            return JsonResponse({'error': 'Invalid JSON in request body'}, status=400)
        except Exception as e:
            # Catch any unexpected errors during the request processing
            logging.exception(f"Fatal error in translate_text view: {e}")
            return JsonResponse({'error': f"Server error: {str(e)}"}, status=500)
    else:
        # Return an error for unsupported HTTP methods
        return JsonResponse({'error': 'Invalid request method. Only POST is supported.'}, status=400)
    
# --- NEW API ENDPOINT VIEWS ---

def api_process_text(request, word):
    """
    API endpoint to process text provided in the URL path and return JSON.
    """
    if request.method == "GET":
        text_to_process = word # Get text directly from the URL path
        results = _get_text_analysis_results(text_to_process)
        return JsonResponse(results)
    else:
        return JsonResponse({'error': 'Method not allowed. Use GET for this API endpoint.'}, status=405)


def api_translate(request, word):
    """
    API endpoint to translate text provided in the URL path to French and return JSON.
    """
    if request.method == "GET":
        text_to_translate = word # Get text directly from the URL path
        target_language = 'fr' # Hardcoded to French as per your original example

        if not translator:
            return JsonResponse({'error': 'Translation service not initialized. Check server logs.'}, status=500)

        try:
            translated = translator.translate(text_to_translate, dest=target_language)
            return JsonResponse({'translated_text': translated.text})
        except Exception as e:
            print(f"Error during googletrans API call for /{word}/: {e}")
            return JsonResponse({'error': f"Translation API Error (googletrans): {e}. This library is unstable."}, status=500)
    else:
        return JsonResponse({'error': 'Method not allowed. Use GET for this API endpoint.'}, status=405)




def translate_word(word, language):
    translator = Translator(to_lang=language)
    translation = translator.translate(word)
    return translation




#AJADI  Chartbot API

@csrf_exempt
def chatbot(request):
    if request.method != 'POST':
        return JsonResponse({'status': 'error', 'response': 'Invalid request method'}, status=405)

    # Safely extract data from form-data
    user_message = request.POST.get('message', '').strip()
    session_key = request.POST.get('session_key', '').strip()
    attachment = request.FILES.get('attachment', None)  # Optional attachment

    if not user_message and not attachment:
        return JsonResponse({'status': 'error', 'response': 'Message or attachment is required'}, status=400)

    if not session_key:
        return JsonResponse({'status': 'error', 'response': 'Session key is required'}, status=400)

    # Get or create conversation session
    conversation, _ = Conversation.objects.get_or_create(session_id=session_key, is_active=True)

    # Create user message
    user_msg_obj = Message.objects.create(
        conversation=conversation,
        text=user_message,
        is_user=True
    )

    # Save attachment if provided
    file_path = ""
    if attachment:
        user_msg_obj.attachment = attachment
        user_msg_obj.save()
        try:
            file_path = user_msg_obj.attachment.path
        except Exception as e:
            logging.warning(f"Could not resolve attachment path: {e}")
            file_path = ""

    # Call your chatbot processor
    try:
        bot_response_data = process_message(user_message, session_key, file_path)
        bot_metadata = bot_response_data.get('metadata', "I'm sorry, I couldn't process your request.")
    except Exception as e:
        bot_metadata = f"Error processing message: {str(e)}"
        logging.error(f"process_message failed: {e}")

    # Craft the response payload
    response_payload = {
        'status': 'success',
        'response': bot_metadata,
        'attachment_url': user_msg_obj.attachment.url if attachment else None
    }

    logging.info(f"Bot response: {bot_metadata}")
    return JsonResponse(response_payload)





@csrf_exempt
def word_process(request):
    """
    API endpoint to process text submitted via form-data
    and return analysis results as JSON.
    """
    if request.method != 'POST':
        return JsonResponse({'error': 'Method not allowed. Use POST for this API endpoint.'}, status=405)

    # Get the 'word' from form-data
    word = request.POST.get('word', '').strip()

    if not word:
        return JsonResponse({'error': 'No text provided for processing'}, status=400)

    try:
        results = _get_text_analysis_results(word)
        return JsonResponse(results)
    except Exception as e:
        return JsonResponse({'error': f'Text processing failed: {str(e)}'}, status=500)


@csrf_exempt
def word_translate(request):
    """
    API endpoint to translate text to a target language.
    Accepts form-data or x-www-form-urlencoded.
    Defaults target language to French ('fr').
    """
    if request.method != 'POST':
        return JsonResponse({'status': 'error', 'response': 'Invalid request method'}, status=405)

    # Extract POST values safely
    word = request.POST.get('word', '').strip()
    target_language = request.POST.get('lang', 'fr').strip()

    if not word:
        return JsonResponse({'error': 'No text provided for translation'}, status=400)

    if not target_language:
        return JsonResponse({'error': 'No target language specified'}, status=400)

    # Confirm translation engine is set
    if 'translator' not in globals() or not translator:
        logging.error("Translation service not initialized.")
        return JsonResponse({'error': 'Translation service not initialized. Check server logs.'}, status=500)

    # Perform translation
    try:
        translated = translator.translate(word, dest=target_language)
        return JsonResponse({'translated_text': translated.text})
    except Exception as e:
        logging.error(f"Translation error for '{word}': {e}")
        return JsonResponse({'error': f"Translation API Error: {str(e)}"}, status=500)
    

@csrf_exempt
def variance2(request):
    print("Variance endpoint called")
    if request.method != 'POST':
     return JsonResponse({'status': 'error', 'response': 'Please submit via POST with two files: "old" and "new".'}, status=405)

    old = request.FILES.get('old')
    new = request.FILES.get('new')

    if not old or not new:
        return JsonResponse({'status': 'error', 'response': 'Missing files'}, status=400)

    key_fields = important()

    try:
        datar = json.load(old)
        datat = json.load(new)

        payslips_dfr = pd.json_normalize(datar["payslips"])
        payslips_dft = pd.json_normalize(datat["payslips"])

        initial_json = payslips_dfr[key_fields].to_json(orient="records", indent=4)
        treated_json = payslips_dft[key_fields].to_json(orient="records", indent=4)

        result = atb(initial_json, treated_json)
        print("Variance endpoint successful")
        return JsonResponse({'status': 'success', 'data': result}, status=200)

    except Exception as e:
        print("Variance endpoint failed")
        return JsonResponse({'status': 'error', 'response': str(e)}, status=500)


    # if not user_message and not attachment:
    #     return JsonResponse({'status': 'error', 'response': 'Message or attachment is required'}, status=400)

    # if not session_key:
    #     return JsonResponse({'status': 'error', 'response': 'Session key is required'}, status=400)

    # # Get or create conversation session
    # conversation, _ = Conversation.objects.get_or_create(session_id=session_key, is_active=True)

    # # Create user message
    # user_msg_obj = Message.objects.create(
    #     conversation=conversation,
    #     text=user_message,
    #     is_user=True
    # )

    # # Save attachment if provided
    # file_path = ""
    # if attachment:
    #     user_msg_obj.attachment = attachment
    #     user_msg_obj.save()
    #     try:
    #         file_path = user_msg_obj.attachment.path
    #     except Exception as e:
    #         logging.warning(f"Could not resolve attachment path: {e}")
    #         file_path = ""

    # # Call your chatbot processor
    # try:
    #     bot_response_data = process_message(user_message, session_key, file_path)
    #     bot_metadata = bot_response_data.get('metadata', "I'm sorry, I couldn't process your request.")
    # except Exception as e:
    #     bot_metadata = f"Error processing message: {str(e)}"
    #     logging.error(f"process_message failed: {e}")

    # # Craft the response payload
    # response_payload = {
    #     'status': 'success',
    #     'response': bot_metadata,
    #     'attachment_url': user_msg_obj.attachment.url if attachment else None
    # }

    # logging.info(f"Bot response: {bot_metadata}")
    # return JsonResponse(response_payload)





from django.http import JsonResponse, HttpResponse
from django.template import loader  # For rendering HTML templates
from django.views.decorators.csrf import csrf_exempt
import json
import pandas as pd

@csrf_exempt
def variance1(request):
    print("Variance endpoint called")

    if request.method != 'POST':
        return JsonResponse({
            'status': 'error',
            'response': 'Please submit via POST with two files: "old" and "new".'
        }, status=405)

    old = request.FILES.get('old')
    new = request.FILES.get('new')

    if not old or not new:
        return JsonResponse({
            'status': 'error',
            'response': 'Missing files'
        }, status=400)

    key_fields = important()

    try:
        datar = json.load(old)
        datat = json.load(new)

        payslips_dfr = pd.json_normalize(datar["payslips"])
        payslips_dft = pd.json_normalize(datat["payslips"])

        initial_json = payslips_dfr[key_fields].to_json(orient="records", indent=4)
        treated_json = payslips_dft[key_fields].to_json(orient="records", indent=4)

        result = atb(initial_json, treated_json)

        # 🎯 Check if HTML is requested (e.g., ?format=html or Accept header)
        if request.GET.get("format") == "html":
            template = loader.get_template('variance_result.html')  # You’ll create this template
            context = {
                "status": "success",
                "data": result
            }
            return HttpResponse(template.render(context, request))

        print("Variance endpoint successful")
        return JsonResponse({'status': 'success', 'data': result}, status=200)

    except Exception as e:
        print("Variance endpoint failed")
        return JsonResponse({
            'status': 'error',
            'response': str(e)
        }, status=500)


from django.shortcuts import render
desired_columns = desire()
@csrf_exempt
def variance(request):
    print("Variance endpoint called")

    if request.method != 'POST':
        return JsonResponse({
            'status': 'error',
            'response': 'Please submit via POST with two files: "old" and "new".'
        }, status=405)

    raw_old = request.FILES.get('old')
    raw_new = request.FILES.get('new')

    if not raw_old or not raw_new:
        return JsonResponse({'status': 'error', 'response': 'Missing files'}, status=400)

    key_fields = important()
    print("Variance endpoint MIDDLE ")
    

    try:
        old = get_payslips_from_json(raw_old,desired_columns)
        new = get_payslips_from_json(raw_new,desired_columns)
        # datar = json.load(old)
        # datat = json.load(new)
       
        # payslips_dfr = pd.json_normalize(datar["payslips"])
        # payslips_dft = pd.json_normalize(datat["payslips"])

        # initial_json = payslips_dfr[key_fields].to_json(orient="records", indent=4)
        # treated_json = payslips_dft[key_fields].to_json(orient="records", indent=4)
        py2 = Prompt7.objects.get(pk=1)  # Get the existing record

        retrieved_template6 = py2.variance_prompt

        result = atb(old, new,llmv,retrieved_template6)

        if request.GET.get("format") == "html":
            print("Variance endpoint successful")
            return render(request, 'myapp/variance_result.html', {
                'status': 'success',
                'data': result
            })
        

        print("Variance endpoint successful")
        return JsonResponse({'status': 'success', 'data': result}, status=200)

    except Exception as e:
        print("Variance endpoint failed")
        return JsonResponse({'status': 'error', 'response': str(e)}, status=500)
    
from django.shortcuts import render

def variance_upload_form(request):
    return render(request, 'myapp/variance_upload.html')  # This matches your file path