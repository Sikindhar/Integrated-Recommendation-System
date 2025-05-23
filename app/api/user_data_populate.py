import pandas as pd
from pymongo import MongoClient
from dotenv import load_dotenv
import os

load_dotenv()

client = MongoClient(os.getenv('MONGODB_URI'))
db = client[os.getenv('DATABASE_NAME')]
users = db['users']

def populate_users_from_csv():
    ratings_df = pd.read_csv('data/processed_ratings.csv')
    
    unique_users = ratings_df['userId'].unique()
    
    for user_id in unique_users:
        if users.find_one({"userId": user_id}) is None:
            users.insert_one({"userId": user_id})
    
    print("Users populated successfully.")

def add_new_user(user_id):
    if users.find_one({"userId": user_id}) is None:
        users.insert_one({"userId": user_id})
        print(f"User {user_id} added successfully.")
    else:
        print(f"User {user_id} already exists.")

if __name__ == "__main__":
    populate_users_from_csv()
