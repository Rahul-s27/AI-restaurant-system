from flask import Flask, render_template, request, jsonify
import pandas as pd
import pickle
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler

app = Flask(__name__)

# Load the models
def load_models():
    with open('rating_model.pkl', 'rb') as f:
        rating_model = pickle.load(f)
    with open('cuisine_model.pkl', 'rb') as f:
        cuisine_model = pickle.load(f)
    with open('recommendation_model.pkl', 'rb') as f:
        recommendation_model = pickle.load(f)
    return rating_model, cuisine_model, recommendation_model

@app.route('/')
def home():
    return render_template('index.html')
@app.route('/search_by_cuisine', methods=['POST'])
def search_by_cuisine():
    try:
        cuisine = request.form['cuisine']
        df = pd.read_csv('Dataset.csv')
        
        # Filter restaurants by cuisine
        matching_restaurants = df[df['Cuisines'].str.contains(cuisine, case=False, na=False)]
        matching_restaurants = matching_restaurants.sort_values(by='Aggregate rating', ascending=False).head(10)
        
        result = []
        for _, row in matching_restaurants.iterrows():
            result.append({
                'name': row['Restaurant Name'],
                'cuisines': row['Cuisines'],
                'rating': row['Aggregate rating'],
                'cost': row['Average Cost for two'],
                'city': row['City'] if 'City' in row else 'Unknown'
            })
            
        return jsonify({'success': True, 'restaurants': result})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})
    

@app.route('/predict_rating', methods=['POST'])
def predict_rating():
    try:
        # Get data from form and validate
        if not all(key in request.form for key in ['cost', 'table_booking', 'online_delivery', 'price_range']):
            return jsonify({'success': False, 'error': 'Missing required fields'})

        # Create input data dictionary with proper data types
        data = {
            'Average Cost for two': float(request.form['cost']),
            'Has Table booking': 1 if request.form['table_booking'].lower() == 'yes' else 0,
            'Has Online delivery': 1 if request.form['online_delivery'].lower() == 'yes' else 0,
            'Price range': int(request.form['price_range'])
        }
        
        # Create DataFrame and validate data types
        input_df = pd.DataFrame([data])
        
        # Scale the numerical features
        scaler = StandardScaler()
        numerical_features = ['Average Cost for two', 'Price range']
        input_df[numerical_features] = scaler.fit_transform(input_df[numerical_features])
        
        # Make prediction
        try:
            prediction = rating_model.predict(input_df)
            predicted_rating = max(0, min(5, round(float(prediction[0]), 2)))  # Ensure rating is between 0 and 5
            return jsonify({'success': True, 'prediction': predicted_rating})
        except Exception as model_error:
            print(f"Prediction error: {str(model_error)}")
            return jsonify({'success': False, 'error': 'Error making prediction'})
            
    except Exception as e:
        print(f"Error in predict_rating: {str(e)}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/recommend', methods=['POST'])
def recommend():
    try:
        # Load restaurant data
        df = pd.read_csv('Dataset.csv')
        
        # Get restaurant name and validate
        if 'restaurant_name' not in request.form:
            return jsonify({'success': False, 'error': 'Restaurant name is required'})
        
        restaurant_name = request.form['restaurant_name'].strip()
        
        # Case-insensitive search for restaurant
        restaurant_mask = df['Restaurant Name'].str.lower() == restaurant_name.lower()
        if not restaurant_mask.any():
            return jsonify({'success': False, 'error': 'Restaurant not found'})
        
        # Get the restaurant index and info
        restaurant_idx = df[restaurant_mask].index[0]
        target_restaurant = df.iloc[restaurant_idx]
        
        # Get target restaurant's cuisines
        target_cuisines = set(cuisine.strip() for cuisine in target_restaurant['Cuisines'].split(','))
        
        # Calculate cuisine similarity for all restaurants
        cuisine_similarities = []
        for _, row in df.iterrows():
            current_cuisines = set(cuisine.strip() for cuisine in str(row['Cuisines']).split(','))
            similarity = len(target_cuisines.intersection(current_cuisines)) / len(target_cuisines.union(current_cuisines)) if current_cuisines else 0
            cuisine_similarities.append(similarity)
        
        # Get top 5 similar restaurants
        similar_indices = sorted(range(len(cuisine_similarities)), 
                               key=lambda i: cuisine_similarities[i], 
                               reverse=True)[1:6]  # Exclude the restaurant itself
        
        recommendations = []
        for idx in similar_indices:
            restaurant = df.iloc[idx]
            recommendations.append({
                'name': restaurant['Restaurant Name'],
                'cuisines': restaurant['Cuisines'],
                'rating': float(restaurant['Aggregate rating']),
                'cost': float(restaurant['Average Cost for two']),
                'similarity': round(cuisine_similarities[idx] * 100, 1),
                'location': restaurant['City'] if 'City' in restaurant else 'Unknown'
            })
        
        return jsonify({
            'success': True,
            'selected_restaurant': {
                'name': target_restaurant['Restaurant Name'],
                'cuisines': target_restaurant['Cuisines'],
                'rating': float(target_restaurant['Aggregate rating']),
                'cost': float(target_restaurant['Average Cost for two']),
                'location': target_restaurant['City'] if 'City' in target_restaurant else 'Unknown'
            },
            'recommendations': recommendations
        })
        
    except Exception as e:
        print(f"Error in recommend: {str(e)}")
        return jsonify({'success': False, 'error': str(e)})
    
if __name__ == '__main__':
    # Load models globally
    rating_model, cuisine_model, recommendation_model = load_models()
    app.run(debug=True)