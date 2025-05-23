from fastapi import FastAPI, Request
import requests
import os
from dotenv import load_dotenv

load_dotenv()

WHATSAPP_TOKEN = os.getenv('WHATSAPP_TOKEN')
WHATSAPP_PHONE_NUMBER_ID = os.getenv('WHATSAPP_PHONE_NUMBER_ID')

def send_whatsapp_message(to_number: str, message: str):
    url = f"https://graph.facebook.com/v22.0/{WHATSAPP_PHONE_NUMBER_ID}/messages"
    
    headers = {
        "Authorization": f"Bearer {WHATSAPP_TOKEN}",
        "Content-Type": "application/json"
    }
    
    data = {
        "messaging_product": "whatsapp",
        "to": to_number,
        "type": "text",
        "text": {"body": message}
    }
    
    response = requests.post(url, headers=headers, json=data)
    return response.json()

def handle_whatsapp_message(message: dict):
    try:
        entry = message['entry'][0]
        changes = entry['changes'][0]
        value = changes['value']
        
        if 'messages' in value:
            message_data = value['messages'][0]
            from_number = message_data['from']
            message_text = message_data['text']['body']
            
            if message_text == "this is a text message":
                return """Test Mode - Available Commands:
1. 'recommendations' - Get product recommendations
2. 'history' - See your rating history
3. 'rate <product_id> <rating>' - Rate a product (1-5)
                
To test in production, send these commands directly to the WhatsApp number."""
            
            if message_text.lower().startswith('recommendations'):
                user_id = from_number  
                try:
                    user_product_matrix = get_user_product_matrix()
                    recommendations = get_recommendations_for_user(user_id, user_product_matrix)
                    return f"Here are your recommendations:\n{format_recommendations(recommendations)}"
                except Exception as e:
                    return "Sorry, I couldn't get recommendations at this time."
                    
            elif message_text.lower().startswith('history'):
                user_id = from_number
                try:
                    history = get_user_history(user_id)
                    return f"Here is your rating history:\n{format_history(history)}"
                except Exception as e:
                    return "Sorry, I couldn't get your history at this time."
                    
            else:
                return """Welcome! Here are the available commands:
1. 'recommendations' - Get product recommendations
2. 'history' - See your rating history
3. 'rate <product_id> <rating>' - Rate a product (1-5)"""

        return None
    except Exception as e:
        print(f"Error handling message: {str(e)}")
        return None

def format_recommendations(recommendations):
    if not recommendations:
        return "No recommendations available."
    return "\n".join([f"Product {prod_id}: {rating:.2f} stars" for prod_id, rating in recommendations])

def format_history(history):
    if not history:
        return "No history available."
    return "\n".join([f"Product {item['productId']}: {item['rating']} stars" for item in history])
