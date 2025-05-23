import pandas as pd
import numpy as np
from typing import Tuple

def load_and_preprocess_data(
    ratings_path: str,
    min_ratings_per_user: int = 5,
    min_ratings_per_product: int = 5,
    max_users: int = 1000,
    max_products: int = 1000
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load and preprocess the ratings data to create a manageable subset.
    
    Args:
        ratings_path (str): Path to the ratings CSV file
        min_ratings_per_user (int): Minimum number of ratings required per user
        min_ratings_per_product (int): Minimum number of ratings required per product
        max_users (int): Maximum number of users to include
        max_products (int): Maximum number of products to include
        
    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: Processed ratings and products dataframes
    """
    print("Loading ratings data...")
    ratings_df = pd.read_csv(
        ratings_path,
        names=['userId', 'productId', 'rating', 'timestamp'],
        header=None
    )
    
    print(f"Original dataset size: {len(ratings_df)} ratings")
    
    user_counts = ratings_df['userId'].value_counts()
    valid_users = user_counts[user_counts >= min_ratings_per_user].index
    ratings_df = ratings_df[ratings_df['userId'].isin(valid_users)]
    
    print(f"After filtering users: {len(ratings_df)} ratings")
    
    product_counts = ratings_df['productId'].value_counts()
    valid_products = product_counts[product_counts >= min_ratings_per_product].index
    ratings_df = ratings_df[ratings_df['productId'].isin(valid_products)]
    
    print(f"After filtering products: {len(ratings_df)} ratings")
    
    top_users = user_counts[valid_users].nlargest(max_users).index
    top_products = product_counts[valid_products].nlargest(max_products).index
    
    ratings_df = ratings_df[
        (ratings_df['userId'].isin(top_users)) &
        (ratings_df['productId'].isin(top_products))
    ]
    
    print(f"Final subset size: {len(ratings_df)} ratings")
    print(f"Number of unique users: {ratings_df['userId'].nunique()}")
    print(f"Number of unique products: {ratings_df['productId'].nunique()}")
    

    products_df = pd.DataFrame({
        'productId': top_products,
        'title': [f"Product {pid}" for pid in top_products],  # Placeholder titles
        'description': [f"Description for product {pid}" for pid in top_products],  # Placeholder descriptions
        'category': ['Electronics'] * len(top_products)  # All products are electronics
    })
    
    return ratings_df, products_df

if __name__ == "__main__":

    ratings_df, products_df = load_and_preprocess_data(
        'app/data/ratings_Electronics.csv',
        min_ratings_per_user=5,
        min_ratings_per_product=5,
        max_users=1000,
        max_products=1000
    )
    

    ratings_df.to_csv('app/data/processed_ratings.csv', index=False)
    products_df.to_csv('app/data/processed_products.csv', index=False)
    
    print("\nData processing complete!")
    print("Processed files saved as:")
    print("- app/data/processed_ratings.csv")
    print("- app/data/processed_products.csv") 