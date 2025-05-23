from fastapi import FastAPI, Request, Query
from pymongo import MongoClient
from dotenv import load_dotenv
import os
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
from app.api.whatsapp_handler import send_whatsapp_message, handle_whatsapp_message
from datetime import datetime

load_dotenv()

app = FastAPI()

client = MongoClient(os.getenv('MONGODB_URI'))
db = client[os.getenv('DATABASE_NAME')]
users = db['users']
products = db['products']
ratings = db['ratings']

def add_new_user(user_id: str):
    if users.find_one({"userId": user_id}) is None:
        users.insert_one({"userId": user_id})
        print(f"User {user_id} added successfully.")
    else:
        print(f"User {user_id} already exists.")

def get_user_product_matrix():
    ratings_data = list(ratings.find())
    
    ratings_df = pd.DataFrame(ratings_data)
    
    user_product_matrix = ratings_df.pivot(index='userId', columns='productId', values='rating').fillna(0)
    
    return user_product_matrix

def get_recommendations_for_user(user_id, user_product_matrix):
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
    
    return predictions  # Return all predictions instead of top_n

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.get("/recommendations/{user_id}")
def get_recommendations(user_id: str):
    try:
        user_product_matrix = get_user_product_matrix()
        recommendations = get_recommendations_for_user(user_id, user_product_matrix)
        return {"user_id": user_id, "recommendations": recommendations}
    except Exception as e:
        return {"error": str(e), "user_id": user_id}

@app.get("/user/{user_id}/history")
def get_user_history(user_id: str):
    try:
        user_ratings = list(ratings.find({"userId": user_id}))
        
        if not user_ratings:
            return {"error": "No history found for this user", "user_id": user_id}
        
        history = []
        for rating in user_ratings:
            product = products.find_one({"productId": rating["productId"]})
            if product:
                history.append({
                    "productId": rating["productId"],
                    "rating": rating["rating"],
                    "productTitle": product.get("title", "Unknown Product"),
                    "timestamp": rating.get("timestamp", "")
                })
        
        return {
            "user_id": user_id,
            "history": history
        }
    except Exception as e:
        return {"error": str(e), "user_id": user_id}

@app.get("/product/{product_id}")
def get_product_details(product_id: str):
    try:
        product = products.find_one({"productId": product_id})
        
        if not product:
            return {"error": "Product not found", "product_id": product_id}
        
        product_ratings = list(ratings.find({"productId": product_id}))
        avg_rating = sum(r["rating"] for r in product_ratings) / len(product_ratings) if product_ratings else 0
        
        return {
            "product_id": product_id,
            "title": product.get("title", "Unknown Product"),
            "category": product.get("category", "Unknown Category"),
            "description": product.get("description", "No description available"),
            "average_rating": round(avg_rating, 2),
            "total_ratings": len(product_ratings)
        }
    except Exception as e:
        return {"error": str(e), "product_id": product_id}

@app.get("/user/{user_id}/preferences")
def get_user_preferences(user_id: str):
    try:
        user_ratings = list(ratings.find({"userId": user_id}))
        
        if not user_ratings:
            return {"error": "No preferences found for this user", "user_id": user_id}
        
        preferences = {
            "user_id": user_id,
            "total_ratings": len(user_ratings),
            "average_rating": 0,
            "preferred_categories": {},
            "rating_distribution": {
                "1": 0, "2": 0, "3": 0, "4": 0, "5": 0
            }
        }
        
        total_rating = 0
        for rating in user_ratings:
            
            rating_value = str(int(rating["rating"]))
            preferences["rating_distribution"][rating_value] += 1
            
            total_rating += rating["rating"]
            
            product = products.find_one({"productId": rating["productId"]})
            if product and "category" in product:
                category = product["category"]
                if category in preferences["preferred_categories"]:
                    preferences["preferred_categories"][category] += 1
                else:
                    preferences["preferred_categories"][category] = 1
        
        preferences["average_rating"] = round(total_rating / len(user_ratings), 2)
        
        preferences["preferred_categories"] = dict(
            sorted(
                preferences["preferred_categories"].items(),
                key=lambda x: x[1],
                reverse=True
            )
        )
        
        return preferences
    except Exception as e:
        return {"error": str(e), "user_id": user_id}

@app.post("/users/")
def create_user(user_id: str):
    add_new_user(user_id)
    return {"message": "User created successfully"}

@app.post("/webhook")
async def webhook(request: Request):
    try:
        print("\n=== Received Webhook Request ===")
        print("Request received at:", datetime.now())
        
        body = await request.json()
        print("Request body:", body)
        
        response = handle_whatsapp_message(body)
        print("Response:", response)
        
        if response:
            from_number = body['entry'][0]['changes'][0]['value']['messages'][0]['from']
            print("Sending response to:", from_number)
            send_whatsapp_message(from_number, response)
        
        return {"status": "success"}
    except Exception as e:
        print("Error in webhook:", str(e))
        print("Error details:", e.__class__.__name__)
        return {"status": "error", "message": str(e)}

@app.get("/webhook")
async def verify_webhook(
    hub_mode: str = Query(None, alias="hub.mode"),
    hub_verify_token: str = Query(None, alias="hub.verify_token"),
    hub_challenge: str = Query(None, alias="hub.challenge")
):
    print("=== Webhook Verification Request ===")
    print(f"hub.mode: {hub_mode}")
    print(f"hub.verify_token: {hub_verify_token}")
    print(f"hub.challenge: {hub_challenge}")
    print(f"Expected token: {os.getenv('WHATSAPP_VERIFY_TOKEN')}")
    print("===================================")
    
    # Check if this is a verification request
    if hub_mode == "subscribe" and hub_verify_token:
        if hub_verify_token == os.getenv('WHATSAPP_VERIFY_TOKEN'):
            print("Verification successful")
            return int(hub_challenge)
        print("Verification failed - token mismatch")
        return "Forbidden", 403
    
    print("Not a verification request")
    return "Bad Request", 400