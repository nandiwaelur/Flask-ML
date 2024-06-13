from flask import Flask, request, jsonify
from predict import load_data, destination_recommendations
import numpy as np

app = Flask(__name__)

# Load data and cosine similarity matrix
data, cosine_sim_df = load_data()

# Route to handle recommendation requests via POST
@app.route('/recommend', methods=['POST'])
def recommend_destinations():
    # Ensure request is a POST method
    if request.method == 'POST':
        # Get destination_name from JSON payload
        request_data = request.get_json()

        if not request_data or 'destination_name' not in request_data:
            return jsonify({'error': 'Missing destination_name in request body'}), 400

        destination_name = request_data['destination_name']

        # Get recommendations based on destination_name
        if destination_name in data["destination_name"].values:
            # Get recommendations for the input destination
            recommendations = destination_recommendations(destination_name, data, cosine_sim_df)
        else:
            # If destination not found, choose a random one and get recommendations
            random_destination = np.random.choice(data["destination_name"].values)
            recommendations = destination_recommendations(random_destination, data, cosine_sim_df)
        
        # Format recommendations into JSON
        json_results = {
            f"recommendation_{i+1}": rec
            for i, rec in enumerate(recommendations)
        }

        return jsonify(json_results)
    else:
        return jsonify({'error': 'Only POST requests are allowed for this endpoint'}), 405


if __name__ == '__main__':
    app.run(debug=True)
