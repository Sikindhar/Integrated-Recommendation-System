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
products = db['new_products']
ratings = db['new_ratings']

def get_user_product_matrix():
    """Create user-product matrix from ratings."""
    ratings_data = list(ratings.find())
    
    ratings_df = pd.DataFrame(ratings_data)
    
    user_product_matrix = ratings_df.pivot(index='userId', columns='productId', values='rating').fillna(0)
    
    return user_product_matrix

def get_recommendations_for_user(user_id, user_product_matrix):
    """Get recommendations using collaborative filtering."""
    try:
        if user_id not in user_product_matrix.index:
            print(f"New user {user_id} - returning popular products")
            all_ratings = list(ratings.find())
            product_ratings = {}
            
            for rating in all_ratings:
                product_id = rating['productId']
                if product_id not in product_ratings:
                    product_ratings[product_id] = {'sum': 0, 'count': 0}
                product_ratings[product_id]['sum'] += rating['rating']
                product_ratings[product_id]['count'] += 1
            
            popular_products = []
            for product_id, stats in product_ratings.items():
                if stats['count'] >= 5:  
                    avg_rating = stats['sum'] / stats['count']
                    popular_products.append((product_id, avg_rating))
            
            popular_products.sort(key=lambda x: x[1], reverse=True)
            return popular_products[:10]  
        
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
            if predicted_rating > 0:
                predictions.append((product, predicted_rating))
        
        predictions.sort(key=lambda x: x[1], reverse=True)
        return predictions[:10]  
        
    except Exception as e:
        print(f"Error in get_recommendations_for_user: {str(e)}")
        return []

def get_action_buttons():
    """Return standard action buttons for navigation."""
    return [
        {
            "type": "actions",
            "elements": [
                {
                    "type": "button",
                    "text": {
                        "type": "plain_text",
                        "text": "📊 Get Recommendations",
                        "emoji": True
                    },
                    "value": "get_recommendations",
                    "action_id": "show_recommendations_input"
                },
                {
                    "type": "button",
                    "text": {
                        "type": "plain_text",
                        "text": "🔍 View Product",
                        "emoji": True
                    },
                    "value": "view_product",
                    "action_id": "show_product_input"
                },
                {
                    "type": "button",
                    "text": {
                        "type": "plain_text",
                        "text": "👤 View User",
                        "emoji": True
                    },
                    "value": "view_user",
                    "action_id": "show_user_input"
                },
                {
                    "type": "button",
                    "text": {
                        "type": "plain_text",
                        "text": "➕ Add New User",
                        "emoji": True
                    },
                    "value": "add_user",
                    "action_id": "show_add_user_input"
                }
            ]
        }
    ]

slack_app = App(token=os.getenv('SLACK_BOT_TOKEN'))

@slack_app.event("message")
def handle_message_events(body, say):
    """Handle any message and show welcome message."""
    try:
        if body.get('bot_id'):
            return
            
        say(blocks=[
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": "🎉 *Welcome to Product Recommendations!*\n\nThis bot helps you discover products based on user preferences. Here's what you can do:"
                }
            },
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": "• Get personalized product recommendations\n• View product details and ratings\n• Check user rating history\n• Add new users to the system"
                }
            },
            {
                "type": "divider"
            },
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": "*Choose an option:*"
                }
            },
            *get_action_buttons()
        ])
    except Exception as e:
        print(f"Error in message handler: {str(e)}")
        say("Sorry, I couldn't process your message. Please try again!")

@slack_app.action("show_recommendations_input")
def handle_recommendations_input(ack, body, say):
    """Show input field for user ID to get recommendations."""
    ack()
    say(blocks=[
        {
            "type": "input",
            "block_id": "user_id_input",
            "element": {
                "type": "plain_text_input",
                "action_id": "user_id",
                "placeholder": {
                    "type": "plain_text",
                    "text": "Enter user ID (e.g., A1F9Z42CFF9IAY)"
                }
            },
            "label": {
                "type": "plain_text",
                "text": "User ID",
                "emoji": True
            }
        },
        {
            "type": "actions",
            "elements": [
                {
                    "type": "button",
                    "text": {
                        "type": "plain_text",
                        "text": "Get Recommendations",
                        "emoji": True
                    },
                    "style": "primary",
                    "value": "get_recommendations",
                    "action_id": "fetch_recommendations"
                }
            ]
        }
    ])

@slack_app.action("show_product_input")
def handle_product_input(ack, body, say):
    """Show input field for product ID."""
    ack()
    say(blocks=[
        {
            "type": "input",
            "block_id": "product_id_input",
            "element": {
                "type": "plain_text_input",
                "action_id": "product_id",
                "placeholder": {
                    "type": "plain_text",
                    "text": "Enter product ID (e.g., 1)"
                }
            },
            "label": {
                "type": "plain_text",
                "text": "Product ID",
                "emoji": True
            }
        },
        {
            "type": "actions",
            "elements": [
                {
                    "type": "button",
                    "text": {
                        "type": "plain_text",
                        "text": "View Product",
                        "emoji": True
                    },
                    "style": "primary",
                    "value": "view_product",
                    "action_id": "fetch_product"
                }
            ]
        }
    ])

@slack_app.action("show_user_input")
def handle_user_input(ack, body, say):
    """Show input field for user ID to view user details."""
    ack()
    say(blocks=[
        {
            "type": "input",
            "block_id": "user_details_input",
            "element": {
                "type": "plain_text_input",
                "action_id": "user_id",
                "placeholder": {
                    "type": "plain_text",
                    "text": "Enter user ID (e.g., 123456789)"
                }
            },
            "label": {
                "type": "plain_text",
                "text": "User ID",
                "emoji": True
            }
        },
        {
            "type": "actions",
            "elements": [
                {
                    "type": "button",
                    "text": {
                        "type": "plain_text",
                        "text": "View User",
                        "emoji": True
                    },
                    "style": "primary",
                    "value": "view_user",
                    "action_id": "fetch_user"
                }
            ]
        }
    ])

@slack_app.action("fetch_recommendations")
def handle_fetch_recommendations(ack, body, say):
    """Handle recommendations request from input."""
    ack()
    try:
        user_id = body['state']['values']['user_id_input']['user_id']['value']
        
        user_product_matrix = get_user_product_matrix()
        recommendations = get_recommendations_for_user(user_id, user_product_matrix)
        
        if not recommendations:
            say(blocks=[
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": f"Sorry, I couldn't find any recommendations for user {user_id}. Please try again later!"
                    }
                },
                *get_action_buttons()
            ])
            return
        
        user = users.find_one({"userId": user_id})
        is_new_user = user is None
        
        header_text = f"🎁 *Top  Recommendations for User {user_id}*"
        if is_new_user:
            header_text += "\n_(Showing popular products for new user)_"
        
        say(blocks=[
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": header_text
                }
            },
            {
                "type": "divider"
            }
        ])
        
        for product_id, predicted_rating in recommendations:
            product = products.find_one({"productId": product_id})
            if product:
                say(blocks=[
                    {
                        "type": "section",
                        "text": {
                            "type": "mrkdwn",
                            "text": f"*{product['title']}*\nCategory: {product.get('category', 'N/A')}\nSimilarity Score: {'⭐' * int(predicted_rating)} ({predicted_rating:.1f})"
                        }
                    },
                    {
                        "type": "divider"
                    }
                ])
        
        say(blocks=get_action_buttons())
        
    except Exception as e:
        print(f"Error in fetch_recommendations: {str(e)}")
        say(blocks=[
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": "Sorry, I couldn't get the recommendations. Please try again!"
                }
            },
            *get_action_buttons()
        ])

@slack_app.action("fetch_product")
def handle_fetch_product(ack, body, say):
    """Handle product details request from input."""
    ack()
    try:
        product_id = body['state']['values']['product_id_input']['product_id']['value']
        print(f"Fetching product with ID: {product_id}")  
        
        product = products.find_one({"productId": product_id})
        print(f"Found product: {product}")  
        
        if not product:
            print(f"No product found with ID: {product_id}")  
            say(blocks=[
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": f"Product {product_id} not found"
                    }
                },
                *get_action_buttons()
            ])
            return
        
        product_ratings = list(ratings.find({"productId": product_id}))
        print(f"Found {len(product_ratings)} ratings for product {product_id}")  # Debug log
        
        avg_rating = sum(r["rating"] for r in product_ratings) / len(product_ratings) if product_ratings else 0
        
        say(blocks=[
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"*Product Details:*\nTitle: {product['title']}\nCategory: {product.get('category', 'N/A')}\nAverage Rating: {'⭐' * int(avg_rating)} ({avg_rating:.1f} stars)\nTotal Ratings: {len(product_ratings)}"
                }
            },
            *get_action_buttons()
        ])
        
    except Exception as e:
        print(f"Error in fetch_product: {str(e)}")
        say(blocks=[
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": "Sorry, I couldn't get the product details. Please try again!"
                }
            },
            *get_action_buttons()
        ])

@slack_app.action("fetch_user")
def handle_fetch_user(ack, body, say):
    """Handle user details request from input."""
    ack()
    try:
        user_id = body['state']['values']['user_details_input']['user_id']['value']
        
        user_ratings = list(ratings.find({"userId": user_id}))
        
        if not user_ratings:
            say(blocks=[
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": f"No ratings found for user {user_id}"
                    }
                },
                *get_action_buttons()
            ])
            return
        
        avg_rating = sum(r["rating"] for r in user_ratings) / len(user_ratings)
        
        recent_ratings_text = "*Recent Ratings:*\n"
        for rating in user_ratings[-5:]:
            product = products.find_one({"productId": rating["productId"]})
            if product:
                stars = "⭐" * int(rating['rating'])  
                recent_ratings_text += f"- {product['title']}: {stars} ({rating['rating']} stars)\n"
        
        say(blocks=[
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"*User Details:*\nUser ID: {user_id}\nTotal Ratings: {len(user_ratings)}\nAverage Rating Given: {'⭐' * int(avg_rating)} ({avg_rating:.1f} stars)"
                }
            },
            {
                "type": "divider"
            },
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": recent_ratings_text
                }
            },
            *get_action_buttons()
        ])
        
    except Exception as e:
        print(f"Error in fetch_user: {str(e)}")
        say(blocks=[
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": "Sorry, I couldn't get the user details. Please try again!"
                }
            },
            *get_action_buttons()
        ])

@slack_app.action("show_add_user_input")
def handle_add_user_input(ack, body, say):
    """Show input fields for new user details."""
    ack()
    say(blocks=[
        {
            "type": "input",
            "block_id": "new_user_id",
            "element": {
                "type": "plain_text_input",
                "action_id": "user_id",
                "placeholder": {
                    "type": "plain_text",
                    "text": "Enter user ID (e.g., A1F9Z42CFF9IAY)"
                }
            },
            "label": {
                "type": "plain_text",
                "text": "User ID",
                "emoji": True
            }
        },
        {
            "type": "input",
            "block_id": "new_user_name",
            "element": {
                "type": "plain_text_input",
                "action_id": "user_name",
                "placeholder": {
                    "type": "plain_text",
                    "text": "Enter user name"
                }
            },
            "label": {
                "type": "plain_text",
                "text": "Name",
                "emoji": True
            }
        },
        {
            "type": "input",
            "block_id": "new_user_email",
            "element": {
                "type": "plain_text_input",
                "action_id": "user_email",
                "placeholder": {
                    "type": "plain_text",
                    "text": "Enter email address"
                }
            },
            "label": {
                "type": "plain_text",
                "text": "Email",
                "emoji": True
            }
        },
        {
            "type": "actions",
            "elements": [
                {
                    "type": "button",
                    "text": {
                        "type": "plain_text",
                        "text": "Create User",
                        "emoji": True
                    },
                    "style": "primary",
                    "value": "create_user",
                    "action_id": "create_new_user"
                }
            ]
        }
    ])

@slack_app.action("create_new_user")
def handle_create_user(ack, body, say):
    """Handle new user creation."""
    ack()
    try:
        values = body['state']['values']
        user_id = values['new_user_id']['user_id']['value']
        user_name = values['new_user_name']['user_name']['value']
        user_email = values['new_user_email']['user_email']['value']
        
        existing_user = users.find_one({"userId": user_id})
        if existing_user:
            say(blocks=[
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": f"❌ User with ID {user_id} already exists!"
                    }
                },
                *get_action_buttons()
            ])
            return
        
        new_user = {
            "userId": user_id,
            "name": user_name,
            "email": user_email,
            "created_at": pd.Timestamp.now()
        }
        
        users.insert_one(new_user)
        
        say(blocks=[
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"✅ Successfully created new user:\n*ID:* {user_id}\n*Name:* {user_name}\n*Email:* {user_email}"
                }
            },
            *get_action_buttons()
        ])
        
    except Exception as e:
        print(f"Error in create_user: {str(e)}")
        say(blocks=[
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": "Sorry, I couldn't create the user. Please try again!"
                }
            },
            *get_action_buttons()
        ])

@app.on_event("startup")
async def startup_event():
    def run_slack_app():
        print("Starting Slack app...")
        try:
            handler = SocketModeHandler(slack_app, os.getenv('SLACK_APP_TOKEN'))
            handler.connect()
            print("Slack app is now running and listening for messages...")
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