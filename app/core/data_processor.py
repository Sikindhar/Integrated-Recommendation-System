import pandas as pd
from typing import Dict, List, Optional, Tuple
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class DataProcessor:
    def __init__(self):
        """Initialize the DataProcessor with empty dataframes."""
        self.ratings_df: Optional[pd.DataFrame] = None
        self.products_df: Optional[pd.DataFrame] = None
        self.tfidf_vectorizer: Optional[TfidfVectorizer] = None
        self.product_vectors: Optional[np.ndarray] = None

    def load_data(self, ratings_path: str, products_path: str) -> None:
        """
        Load ratings and products data from CSV files.
        
        Args:
            ratings_path (str): Path to ratings CSV file
            products_path (str): Path to products CSV file
        """
        self.ratings_df = pd.read_csv(ratings_path)
        self.products_df = pd.read_csv(products_path)
        self._preprocess_data()

    def _preprocess_data(self) -> None:
        """Preprocess the loaded data."""
        if self.products_df is not None:
            self.products_df['text_features'] = self.products_df['title'] + ' ' + self.products_df['description']
            
            self.tfidf_vectorizer = TfidfVectorizer(
                stop_words='english',
                max_features=5000,
                ngram_range=(1, 2)
            )
            
            self.product_vectors = self.tfidf_vectorizer.fit_transform(
                self.products_df['text_features']
            )

    def get_product_details(self, product_id: str) -> Dict:
        """
        Get details for a specific product.
        
        Args:
            product_id (str): Product ID
            
        Returns:
            Dict: Product details
        """
        if self.products_df is None:
            return {}
        
        product = self.products_df[self.products_df['productId'] == product_id]
        if product.empty:
            return {}
        
        return product.iloc[0].to_dict()

    def get_user_ratings(self, user_id: str) -> List[Tuple[str, float]]:
        """
        Get all ratings for a specific user.
        
        Args:
            user_id (str): User ID
            
        Returns:
            List[Tuple[str, float]]: List of (product_id, rating) tuples
        """
        if self.ratings_df is None:
            return []
        
        user_ratings = self.ratings_df[self.ratings_df['userId'] == user_id]
        return list(zip(user_ratings['productId'], user_ratings['rating']))

    def get_product_similarity(self, product_id: str, n_similar: int = 5) -> List[Tuple[str, float]]:
        """
        Get similar products based on content similarity.
        
        Args:
            product_id (str): Product ID
            n_similar (int): Number of similar products to return
            
        Returns:
            List[Tuple[str, float]]: List of (product_id, similarity_score) tuples
        """
        if self.product_vectors is None or self.products_df is None:
            return []
        
        product_idx = self.products_df[self.products_df['productId'] == product_id].index
        if len(product_idx) == 0:
            return []
        
        product_vector = self.product_vectors[product_idx[0]]
        similarity_scores = cosine_similarity(product_vector, self.product_vectors).flatten()
        
        similar_indices = similarity_scores.argsort()[::-1][1:n_similar+1]
        similar_products = [
            (self.products_df.iloc[idx]['productId'], float(similarity_scores[idx]))
            for idx in similar_indices
        ]
        
        return similar_products

    def get_user_product_matrix(self) -> pd.DataFrame:
        """
        Create a user-product rating matrix.
        
        Returns:
            pd.DataFrame: User-product rating matrix
        """
        if self.ratings_df is None:
            return pd.DataFrame()
        
        return self.ratings_df.pivot(
            index='userId',
            columns='productId',
            values='rating'
        ).fillna(0) 