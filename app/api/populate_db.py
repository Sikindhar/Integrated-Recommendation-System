import pandas as pd
from pymongo import MongoClient
from dotenv import load_dotenv
import os

load_dotenv()

client = MongoClient(os.getenv('MONGODB_URI'))
db = client[os.getenv('DATABASE_NAME')]
users = db['users']
products = db['products']
ratings = db['ratings']

ratings_df = pd.read_csv('data/processed_ratings.csv')
products_df = pd.read_csv('data/processed_products.csv')

ratings_records = ratings_df.to_dict('records')
products_records = products_df.to_dict('records')

ratings.insert_many(ratings_records)
products.insert_many(products_records)

ratings.create_index([('userId', 1), ('productId', 1)], unique=True)