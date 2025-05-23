from fastapi import FastAPI
from pymongo import MongoClient
from dotenv import load_dotenv
import os
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

# Load environment variables from .env file
load_dotenv()

app = FastAPI()

# Connect to MongoDB using environment variables
client = MongoClient(os.getenv('MONGODB_URI'))
db = client[os.getenv('DATABASE_NAME')]
users = db['users']
products = db['products']
ratings = db['ratings']

def get_user_product_matrix():
    # Fetch all ratings from MongoDB
    ratings_data = list(ratings.find())
    
    # Convert to DataFrame
    ratings_df = pd.DataFrame(ratings_data)
    
    # Pivot to create user-product matrix
    user_product_matrix = ratings_df.pivot(index='userId', columns='productId', values='rating').fillna(0)
    
    return user_product_matrix

def get_recommendations_for_user(user_id, user_product_matrix, top_n=5):
    # Compute user similarity
    user_similarity = cosine_similarity(user_product_matrix)
    
    # Get the index of the user
    user_idx = user_product_matrix.index.get_loc(user_id)
    
    # Get similar users
    similar_users = user_similarity[user_idx].argsort()[::-1][1:11]  # Top 10 similar users
    
    # Get products the user hasn't rated
    user_ratings = user_product_matrix.iloc[user_idx]
    unrated_products = user_ratings[user_ratings == 0].index
    
    # Predict ratings for unrated products
    predictions = []
    for product in unrated_products:
        product_idx = user_product_matrix.columns.get_loc(product)
        similar_user_ratings = user_product_matrix.iloc[similar_users, product_idx]
        predicted_rating = similar_user_ratings.mean()
        predictions.append((product, predicted_rating))
    
    # Sort predictions by rating
    predictions.sort(key=lambda x: x[1], reverse=True)
    
    return predictions[:top_n]

@app.get("/recommendations/{user_id}")
def get_recommendations(user_id: str):
    user_product_matrix = get_user_product_matrix()
    recommendations = get_recommendations_for_user(user_id, user_product_matrix)
    return {"user_id": user_id, "recommendations": recommendations}

@app.get("/user/{user_id}/history")
def get_user_history(user_id: str):
    try:
        # Get user's ratings from MongoDB
        user_ratings = list(ratings.find({"userId": user_id}))
        
        if not user_ratings:
            return {"error": "No history found for this user", "user_id": user_id}
        
        # Get product details for each rating
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
        # Get product details from MongoDB
        product = products.find_one({"productId": product_id})
        
        if not product:
            return {"error": "Product not found", "product_id": product_id}
        
        # Get average rating for the product
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
        # Get user's ratings
        user_ratings = list(ratings.find({"userId": user_id}))
        
        if not user_ratings:
            return {"error": "No preferences found for this user", "user_id": user_id}
        
        # Get product details for each rating
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
            # Update rating distribution
            rating_value = str(int(rating["rating"]))
            preferences["rating_distribution"][rating_value] += 1
            
            # Calculate total rating
            total_rating += rating["rating"]
            
            # Get product category
            product = products.find_one({"productId": rating["productId"]})
            if product and "category" in product:
                category = product["category"]
                if category in preferences["preferred_categories"]:
                    preferences["preferred_categories"][category] += 1
                else:
                    preferences["preferred_categories"][category] = 1
        
        # Calculate average rating
        preferences["average_rating"] = round(total_rating / len(user_ratings), 2)
        
        # Sort preferred categories by count
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