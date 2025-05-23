import pandas as pd
from pymongo import MongoClient
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Connect to MongoDB using environment variables
client = MongoClient(os.getenv('MONGODB_URI'))
db = client[os.getenv('DATABASE_NAME')]
users = db['users']
products = db['products']
ratings = db['ratings']

# Load data from CSV
ratings_df = pd.read_csv('data/processed_ratings.csv')
products_df = pd.read_csv('data/processed_products.csv')
# Insert data into MongoDB
ratings_records = ratings_df.to_dict('records')
products_records = products_df.to_dict('records')

ratings.insert_many(ratings_records)
products.insert_many(products_records)

# Create a unique index on user_id and product_id in the ratings collection
ratings.create_index([('userId', 1), ('productId', 1)], unique=True)