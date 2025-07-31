# myapp/utils.py
import requests
import os
import logging
from django.conf import settings

logger = logging.getLogger(__name__)

# Base URL for WhatsApp Business API
# Make sure settings.WHATSAPP_API_VERSION is correctly defined in your settings.py
WHATSAPP_API_BASE_URL = f"https://graph.facebook.com/{settings.WHATSAPP_API_VERSION}"


def send_whatsapp_message(to_number, message_body):
    """
    Sends a text message to a WhatsApp number using the WhatsApp Business API.
    """
    url = f"{WHATSAPP_API_BASE_URL}/{settings.WHATSAPP_PHONE_NUMBER_ID}/messages"
    headers = {
        "Authorization": f"Bearer {settings.WHATSAPP_ACCESS_TOKEN}",
        "Content-Type": "application/json",
    }
    payload = {
        "messaging_product": "whatsapp",
        "to": to_number,
        "type": "text",
        "text": {"body": message_body},
    }

    try:
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()  # Raises HTTPError for bad responses (4xx or 5xx)
        logger.info(f"Message sent successfully to {to_number}.")
        logger.debug(f"WhatsApp API response: {response.json()}")
        return True
    except requests.exceptions.HTTPError as http_err:
        logger.error(f"HTTP error occurred while sending message: {http_err} - {response.text}", exc_info=True)
        return False
    except requests.exceptions.ConnectionError as conn_err:
        logger.error(f"Connection error occurred while sending message: {conn_err}", exc_info=True)
        return False
    except requests.exceptions.Timeout as timeout_err:
        logger.error(f"Timeout error occurred while sending message: {timeout_err}", exc_info=True)
        return False
    except requests.exceptions.RequestException as req_err:
        logger.error(f"An error occurred while sending message: {req_err}", exc_info=True)
        return False
    except Exception as e:
        logger.error(f"An unexpected error occurred in send_whatsapp_message: {e}", exc_info=True)
        return False


def download_whatsapp_media(media_id, access_token, api_version="v19.0"):
    """
    Downloads media (image, video, etc.) from WhatsApp's servers.

    Args:
        media_id (str): The ID of the media to download.
        access_token (str): Your WhatsApp Business API access token.
        api_version (str): The Graph API version to use (e.g., "v19.0").

    Returns:
        tuple: A tuple containing (file_path, mime_type) if successful, None otherwise.
    """
    media_url = f"https://graph.facebook.com/{api_version}/{media_id}"
    headers = {
        "Authorization": f"Bearer {access_token}"
    }

    try:
        # First, get media metadata (like URL and mime type)
        response = requests.get(media_url, headers=headers)
        response.raise_for_status()
        media_metadata = response.json()
        
        actual_media_url = media_metadata.get('url')
        mime_type = media_metadata.get('mime_type')
        file_name = media_metadata.get('file_name', f"{media_id}.{mime_type.split('/')[-1] if '/' in mime_type else 'file'}") # Simple filename based on ID and type

        if not actual_media_url:
            logger.error(f"No media URL found for media_id: {media_id}")
            return None

        # Determine file extension based on mime_type
        if mime_type and '/' in mime_type:
            extension = mime_type.split('/')[-1]
            if ';' in extension: # Handle cases like 'image/jpeg; charset=binary'
                extension = extension.split(';')[0]
            file_name_base, _ = os.path.splitext(file_name)
            file_name = f"{file_name_base}.{extension}"
        
        # Create a temporary directory if it doesn't exist
        temp_dir = "temp_media"
        os.makedirs(temp_dir, exist_ok=True)
        file_path = os.path.join(temp_dir, file_name)

        # Download the actual media content
        media_content_response = requests.get(actual_media_url, headers=headers, stream=True)
        media_content_response.raise_for_status()

        with open(file_path, 'wb') as f:
            for chunk in media_content_response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        logger.info(f"Media {media_id} downloaded successfully to {file_path}")
        return file_path, mime_type

    except requests.exceptions.HTTPError as http_err:
        logger.error(f"HTTP error downloading media {media_id}: {http_err} - {response.text if 'response' in locals() else 'No response'}", exc_info=True)
        return None
    except requests.exceptions.RequestException as req_err:
        logger.error(f"Request error downloading media {media_id}: {req_err}", exc_info=True)
        return None
    except Exception as e:
        logger.error(f"An unexpected error occurred in download_whatsapp_media for {media_id}: {e}", exc_info=True)
        return None