import pandas as pd
from pymongo import MongoClient
from dotenv import load_dotenv
import os

load_dotenv()

client = MongoClient(os.getenv('MONGODB_URI'))
db = client[os.getenv('DATABASE_NAME')]
new_users = db['new_users']
new_products = db['new_products']
new_ratings = db['new_ratings']

ratings_df = pd.read_csv('app/data/processed_ratings.csv')
products_df = pd.read_csv('app/data/processed_products.csv')

ratings_records = ratings_df.to_dict('records')
products_records = products_df.to_dict('records')

new_ratings.insert_many(ratings_records)
new_products.insert_many(products_records)

new_ratings.create_index([('userId', 1), ('productId', 1)], unique=True)