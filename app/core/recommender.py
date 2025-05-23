from typing import List, Tuple, Dict
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from app.core.data_processor import DataProcessor

class Recommender:
    def __init__(self, data_processor: DataProcessor):
        """
        Initialize the recommender with a data processor.
        
        Args:
            data_processor (DataProcessor): Instance of DataProcessor
        """
        self.data_processor = data_processor
        self.user_product_matrix = None
        self.user_similarity_matrix = None

    def initialize_matrices(self) -> None:
        """Initialize user-product and user-similarity matrices."""
        print("Initializing recommendation matrices...")
        self.user_product_matrix = self.data_processor.get_user_product_matrix()
        print(f"User-product matrix shape: {self.user_product_matrix.shape}")
        
        if not self.user_product_matrix.empty:
            self.user_similarity_matrix = cosine_similarity(self.user_product_matrix)
            print(f"User similarity matrix shape: {self.user_similarity_matrix.shape}")
        else:
            print("Warning: User-product matrix is empty!")

    def get_recommendations(self, user_id: str, n_recommendations: int = 5) -> List[Dict]:
        """
        Get personalized recommendations for a user based on collaborative filtering.
        
        Args:
            user_id (str): User ID
            n_recommendations (int): Number of recommendations to return
            
        Returns:
            List[Dict]: List of recommended products with scores and explanations
        """
        if self.user_product_matrix is None or self.user_similarity_matrix is None:
            print("Warning: Matrices not initialized!")
            return []

        try:
            user_idx = self.user_product_matrix.index.get_loc(user_id)
            print(f"Found user {user_id} at index {user_idx}")
        except KeyError:
            print(f"Warning: User {user_id} not found in matrix!")
            return []

        user_similarities = self.user_similarity_matrix[user_idx]
        similar_users = np.argsort(user_similarities)[::-1][1:11]  
        print(f"Found {len(similar_users)} similar users")

        similar_users_ratings = self.user_product_matrix.iloc[similar_users]
        user_ratings = self.user_product_matrix.iloc[user_idx]

        unrated_products = user_ratings[user_ratings == 0].index
        print(f"Found {len(unrated_products)} unrated products")

        predictions = []
        for product in unrated_products:
            product_ratings = similar_users_ratings[product]
            if product_ratings.sum() > 0:
                pred_rating = np.average(
                    product_ratings,
                    weights=user_similarities[similar_users]
                )
                predictions.append({
                    'product_id': product,
                    'score': float(pred_rating),
                    'explanation': self._generate_explanation(
                        user_id, product, similar_users, user_similarities[similar_users]
                    )
                })

        predictions.sort(key=lambda x: x['score'], reverse=True)
        print(f"Generated {len(predictions)} predictions")
        return predictions[:n_recommendations]

    def _generate_explanation(self, user_id: str, product_id: str, 
                            similar_users: np.ndarray, similarities: np.ndarray) -> str:
        """Generate explanation for why a product was recommended."""
        product_ratings = self.user_product_matrix.iloc[similar_users][product_id]
        
        high_ratings = (product_ratings >= 4).sum()
        
        if high_ratings > 0:
            return f"Recommended because {high_ratings} similar users rated this product highly"
        else:
            return "Recommended based on similar users' preferences"

    def get_user_history(self, user_id: str) -> List[Dict]:
        """
        Get user's rating history.
        
        Args:
            user_id (str): User ID
            
        Returns:
            List[Dict]: List of products rated by the user
        """
        if self.user_product_matrix is None:
            print("Warning: User-product matrix not initialized!")
            return []

        try:
            user_ratings = self.user_product_matrix.loc[user_id]
            rated_products = user_ratings[user_ratings > 0]
            print(f"Found {len(rated_products)} rated products for user {user_id}")
            
            return [
                {
                    'product_id': product_id,
                    'rating': float(rating)
                }
                for product_id, rating in rated_products.items()
            ]
        except KeyError:
            print(f"Warning: User {user_id} not found in matrix!")
            return []

    def get_item_based_recommendations(self, product_id: str, n_recommendations: int = 5) -> List[Tuple[str, float]]:
        """
        Get similar products based on item-based collaborative filtering.
        
        Args:
            product_id (str): Product ID
            n_recommendations (int): Number of recommendations to return
            
        Returns:
            List[Tuple[str, float]]: List of (product_id, score) tuples
        """
        return self.data_processor.get_product_similarity(product_id, n_recommendations) 