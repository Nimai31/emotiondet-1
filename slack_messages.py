from slack_sdk import WebClient
import os

from dotenv import load_dotenv
load_dotenv()

SLACK_BOT_TOKEN = os.getenv("SLACK_API_TOKEN")

client = WebClient(token=SLACK_BOT_TOKEN)

def send_file_to_user(user_id, file_path, message):
    try:
        # Open a DM channel with the user
        im_response = client.conversations_open(users=[user_id])
        channel_id = im_response['channel']['id']

        # upload the file to the DM channel
        response = client.files_upload_v2(
            channel=channel_id,  
            file=file_path,
            initial_comment=message,
        )
        return response
    except Exception as e:
        print(f"Error sending file: {e}")
        return None
    
def send_text_to_user(user_id, message):
    try:
        # Open a DM channel with the user
        im_response = client.conversations_open(users=[user_id])
        channel_id = im_response['channel']['id']

        # Send plain text message
        response = client.chat_postMessage(
            channel=channel_id,
            text=message
        )
        return response
    except Exception as e:
        print(f"Error sending message: {e}")
        return None

def send_face_with_caption(user_id, face_path, caption):
    try:
        # Open a DM channel with the user
        im_response = client.conversations_open(users=[user_id])
        channel_id = im_response['channel']['id']

        response = client.files_upload_v2(
            channel=channel_id,
            file=face_path,
            initial_comment=caption,
        )
        return response
    except Exception as e:
        print(f"Error sending face with caption: {e}")
        return None
