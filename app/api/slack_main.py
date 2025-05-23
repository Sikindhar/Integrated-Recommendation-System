from fastapi import FastAPI, Request
from pymongo import MongoClient
from dotenv import load_dotenv
import os
from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler
import threading
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

load_dotenv()

app = FastAPI()

client = MongoClient(os.getenv('MONGODB_URI'))
db = client[os.getenv('DATABASE_NAME')]
users = db['users']
products = db['products']
ratings = db['ratings']

def get_user_product_matrix():
    """Create user-product matrix from ratings."""
    ratings_data = list(ratings.find())
    
    ratings_df = pd.DataFrame(ratings_data)
    
    user_product_matrix = ratings_df.pivot(index='userId', columns='productId', values='rating').fillna(0)
    
    return user_product_matrix

def get_recommendations_for_user(user_id, user_product_matrix):
    """Get recommendations using collaborative filtering."""
    try:
        user_similarity = cosine_similarity(user_product_matrix)
        
        user_idx = user_product_matrix.index.get_loc(user_id)
        
        similar_users = user_similarity[user_idx].argsort()[::-1][1:11] 
        
        user_ratings = user_product_matrix.iloc[user_idx]
        unrated_products = user_ratings[user_ratings == 0].index
        
        predictions = []
        for product in unrated_products:
            product_idx = user_product_matrix.columns.get_loc(product)
            similar_user_ratings = user_product_matrix.iloc[similar_users, product_idx]
            predicted_rating = similar_user_ratings.mean()
            predictions.append((product, predicted_rating))
        
        predictions.sort(key=lambda x: x[1], reverse=True)
        
        return predictions
    except Exception as e:
        print(f"Error in get_recommendations_for_user: {str(e)}")
        return []

slack_app = App(token=os.getenv('SLACK_BOT_TOKEN'))

@slack_app.command("/start")
def start_command(ack, body, say):
    """Send welcome message with available commands."""
    ack()
    welcome_message = """Welcome to the Product Recommendation Bot! 🎉

Available commands:
/recommendations <user_id> - Get recommendations for a user
/product <product_id> - View product ratings
/user <user_id> - View user details

Examples:
/recommendations 123456789
/product 1
/user 123456789"""
    
    say(text=welcome_message)

@slack_app.command("/recommendations")
def recommendations_command(ack, body, say):
    """Get recommendations for a user."""
    ack()
    try:
        user_id = body['text'].strip()
        if not user_id:
            say("Please provide a user_id. Example: /recommendations 123456789")
            return
        
        user_product_matrix = get_user_product_matrix()
        recommendations = get_recommendations_for_user(user_id, user_product_matrix)
        
        if not recommendations:
            say(f"No recommendations found for user {user_id}")
            return
        
        response = f"📊 Recommended Products for user {user_id}:\n\n"
        
        for product_id, predicted_rating in recommendations:
            product = products.find_one({"productId": product_id})
            if product:
                response += f"🔹 Product: {product['title']}\n"
                response += f"   Category: {product.get('category', 'N/A')}\n"
                response += f"   Predicted Rating: {predicted_rating:.2f} ⭐\n"
                response += f"   Product ID: {product['productId']}\n\n"
        
        response += f"Total Recommendations: {len(recommendations)}"
        
        say(text=response)
    except Exception as e:
        say("Sorry, I couldn't get the recommendations.")

@slack_app.command("/product")
def product_command(ack, body, say):
    """Get product details and ratings."""
    ack()
    try:
        product_id = body['text'].strip()
        if not product_id:
            say("Please provide a product_id. Example: /product 1")
            return
        
        product = products.find_one({"productId": product_id})
        if not product:
            say(f"Product {product_id} not found")
            return
        
        product_ratings = list(ratings.find({"productId": product_id}))
        
        avg_rating = sum(r["rating"] for r in product_ratings) / len(product_ratings) if product_ratings else 0
        
        response = f"""Product Details:
Title: {product['title']}
Category: {product['category']}
Average Rating: {avg_rating:.1f} stars
Total Ratings: {len(product_ratings)}"""
        
        say(text=response)
    except Exception as e:
        say("Sorry, I couldn't get the product details.")

@slack_app.command("/user")
def user_command(ack, body, say):
    """Get user details and rating history."""
    ack()
    try:
        user_id = body['text'].strip()
        if not user_id:
            say("Please provide a user_id. Example: /user 123456789")
            return
        
        user_ratings = list(ratings.find({"userId": user_id}))
        
        if not user_ratings:
            say(f"No ratings found for user {user_id}")
            return
        
        avg_rating = sum(r["rating"] for r in user_ratings) / len(user_ratings)
        
        response = f"""User Details:
User ID: {user_id}
Total Ratings: {len(user_ratings)}
Average Rating Given: {avg_rating:.1f} stars

Recent Ratings:"""
        
        for rating in user_ratings[-5:]:
            product = products.find_one({"productId": rating["productId"]})
            if product:
                response += f"\n- {product['title']}: {rating['rating']} stars"
        
        say(text=response)
    except Exception as e:
        say("Sorry, I couldn't get the user details.")


@app.on_event("startup")
async def startup_event():
    def run_slack_app():
        print("Starting Slack app...")
        try:
            handler = SocketModeHandler(slack_app, os.getenv('SLACK_APP_TOKEN'))
            handler.connect()
            print("Slack app is now running and listening for messages...")
            # Keep the thread alive
            while True:
                import time
                time.sleep(1)
        except Exception as e:
            print(f"Error in Slack app: {str(e)}")
    
    print("Starting Slack app in background thread...")
    slack_thread = threading.Thread(target=run_slack_app)
    slack_thread.daemon = True
    slack_thread.start()
    print("Slack app thread started")

@app.get("/")
def read_root():
    return {"status": "Slack Recommendation Bot is running"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001) 