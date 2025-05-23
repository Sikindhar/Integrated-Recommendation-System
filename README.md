# Integrated-Recommendation-System

A collaborative filtering-based recommendation system with both Slack and WhatsApp interfaces.

## Features

- **Collaborative Filtering**: Uses cosine similarity to find similar users and predict ratings
- **Multiple Interfaces**:
  - Slack Bot (Live & Working)
  - WhatsApp Bot (API Ready, Needs publishing app after webhook setup to actually work real time)
- **Real-time Recommendations**: Get personalized product recommendations
- **User History**: View rating history and preferences
- **Product Details**: Get detailed product information and ratings

## Tech Stack

- **Backend**: FastAPI
- **Database**: MongoDB
- **ML Libraries**: scikit-learn, pandas, numpy
- **Chat Platforms**: Slack Bolt, WhatsApp Business API

## Setup

1. **Environment Variables**:
```env
MONGODB_URI=mongodb://localhost:27017
DATABASE_NAME=recommendation_db
SLACK_BOT_TOKEN=xoxb-your-bot-token
SLACK_APP_TOKEN=xapp-your-app-token
WHATSAPP_TOKEN=your-whatsapp-token
WHATSAPP_PHONE_NUMBER_ID=your-phone-number-id
WHATSAPP_VERIFY_TOKEN=your-verify-token
```

2. **Install Dependencies**:
```bash
pip install -r requirements.txt
```

3. **Run the Server**:
```bash
uvicorn app.api.main:app --reload --port 8001
```

## Dataset & Processing

The system uses a product ratings dataset from Amazon Electronics Dataset with:
- User ratings (1-5 stars)
- Product details (title, category, description)
- User preferences and history

Data Processing Steps:
1. Load data into MongoDB collections - used only a chunk
2. Create user-product matrix
3. Calculate user similarities using cosine similarity
4. Generate recommendations based on similar users' ratings

## API Endpoints

### Slack Commands
- `/start` - Get started with the bot
- `/recommendations <user_id>` - Get personalized recommendations
- `/product <product_id>` - View product details
- `/user <user_id>` - View user details and history

### REST API Endpoints
- `GET /recommendations/{user_id}` - Get recommendations
- `GET /user/{user_id}/history` - Get user rating history
- `GET /product/{product_id}` - Get product details
- `GET /user/{user_id}/preferences` - Get user preferences

## WhatsApp Integration

The WhatsApp bot is API-ready but requires deployment to be fully functional. To test:
1. Set up WhatsApp Business API credentials
2. Deploy the application
3. Configure webhook URL
4. Test with WhatsApp Business API

## Testing

1. **Slack Bot**:
   - Add the bot to your workspace
   - Use slash commands to interact
   - Test with real user IDs

2. **API Testing**:
   - Use Postman or curl to test endpoints
   - Example: ` http://localhost:8001/recommendations/userId`

## Future Enhancements

- Add more recommendation algorithms
- Implement user authentication
- Add product search functionality
- Enhance data visualization
- Add rating analytics

## Contributing

Feel free to submit issues and enhancement requests! 