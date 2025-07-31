# myproject/myapp/urls.py
from django.urls import path
from .views import *

urlpatterns = [
     path('', chat_home, name='chat_home'),
     path('analytics/', chat_home2, name='chat_home2'),
     path('analytics/send/', send_message2, name='send_message_analytics'), 
    path('summary/', summary, name='summary'),
    path('api/<str:param_name>/<str:session_key>/', send_message3, name='send_message3'),
    path('send/', send_message, name='send_message'),
    path('history/', chat_history, name='chat_history'),
    path('conversation/<int:conversation_id>/', view_conversation, name='view_conversation'),
    
    # Add chart display URL
    path('chart/', chart_view, name='chart_view'),
    
    path("register/", register, name="register"),
    path("setup-password/<int:user_id>/<str:token>/", setup_password, name="setup_password"),
    path("password-reset/", password_reset_request, name="password_reset"),
    path("change-password/", change_password, name="change_password"),
    
    
   
    path("login/", user_login, name="login"),
    path("logout/", user_logout, name="logout"),
    
     path("about", about, name="about"),
  path("contact", contact, name="contact"),
    path("blog/", home, name="home"),
    
    
        path('blog/post', post_list, name='post_list'),
    path('blog/homey', homey, name='homey'),
   
    
    path('blog/edit_currency/', edit_currency, name='edit_currency'),

    
    path("blog/dashboard/", dashboard, name="dashboard"),
    #  path('custom_logout/', custom_logout, name='custom_logout'),
    
    
    
        path("blog/list/", prompt_list, name='prompt_list'),
    path('update/<int:prompt_id>/', update_prompt, name='update_prompt'),



    path("blog/clientlistt/", client_list, name="client_list"),
    path("create/", create_client, name="create_client"),
    path("update/<int:client_id>/", update_client, name="update_client"),


    ## WHATSAPP
    
   path('webhook/', whatsapp_webhook, name='whatsapp_webhook'),


   #NP
   path('editor/', editor_view, name='editor'),
    path('process_text/', process_text, name='process_text'), # Existing view for editor AJAX/HTML response
    path('translate/', translate_text, name='translate'),     # Existing view for editor AJAX translation


    
    path('api_process_text/<str:word>/', api_process_text, name='api_process_text'),
    # Translates text from URL path and returns JSON
    path('api_translate/<str:word>/', api_translate, name='api_translate'),
    # --- END NEW API ENDPOINTS ---

    # --- NEW API ENDPOINTS ---
    # Processes text from URL path and returns JSON
    path('chatbot/', chatbot, name='chatbot'),
    # Translates text from URL path and returns JSON
    path('word_translate/', word_translate, name='word_translate'),
    # --- END NEW API ENDPOINTS ---
     path('word_process/', word_process, name='word_process'),

    path('variance/', variance, name='variance'),  # New endpoint for variance calculation

    path('variance_upload/', variance_upload_form, name='variance_upload_form'),
]
    # path('variance/', views.variance_view, name='variance')
 
    




# from django.urls import path
# from .views import prompt_list, update_prompt

# urlpatterns = [
#     path('', prompt_list, name='prompt_list'),
#     path('update/<int:prompt_id>/', update_prompt, name='update_prompt'),
