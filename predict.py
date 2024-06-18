import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import json


# Function to load and preprocess data
def load_data():
    # Load Dataset
    url_tour = "https://raw.githubusercontent.com/Vinzzztty/playground-data-analyst/main/Dataset/pariwisata_jogja/tour2.csv"
    url_rating = "https://raw.githubusercontent.com/Vinzzztty/playground-data-analyst/main/Dataset/pariwisata_jogja/tour_rating.csv"
    url_user = "https://raw.githubusercontent.com/Vinzzztty/playground-data-analyst/main/Dataset/pariwisata_jogja/user.csv"

    tour = pd.read_csv(url_tour)
    rating = pd.read_csv(url_rating)
    user = pd.read_csv(url_user)

    # Merge all Place_Id from tour and rating datasets
    tour_all = np.concatenate((tour.Place_Id.unique(), rating.Place_Id.unique()))
    tour_all = np.sort(np.unique(tour_all))

    # Merge all User_Id from user and rating datasets
    user_all = np.concatenate((user.User_Id.unique(), rating.User_Id.unique()))
    user_all = np.sort(np.unique(user_all))

    # Merge tour_rating with tour dataset based on Place_Id
    all_tour_rate = rating
    all_tour = pd.merge(
        all_tour_rate,
        tour[
            [
                "Place_Id",
                "Place_Name",
                "Category",
                "Rating",
                "Price",
                "Description",
                "Image",
            ]
        ],
        on="Place_Id",
        how="left",
    )

    # Remove duplicates based on Place_Id
    preparation = all_tour.drop_duplicates("Place_Id")

    # Create dataframe for destination details
    destination_id = preparation["Place_Id"].tolist()
    destination_name = preparation["Place_Name"].tolist()
    destination_category = preparation["Category"].tolist()
    destination_price = preparation["Price"].tolist()
    destination_description = preparation["Description"].tolist()
    destination_rating = preparation["Rating"].tolist()
    destination_image = preparation["Image"].tolist()

    data = pd.DataFrame(
        {
            "id": destination_id,
            "destination_name": destination_name,
            "category": destination_category,
            "price": destination_price,
            "description": destination_description,
            "rating": destination_rating,
            "image": destination_image,
        }
    )

    # Initialize TFIDFVectorizer and fit on destination categories
    tf = TfidfVectorizer()
    tfidf_matrix = tf.fit_transform(data["category"])

    # Compute cosine similarity matrix
    cosine_sim = cosine_similarity(tfidf_matrix)

    # Create dataframe from cosine similarity matrix
    cosine_sim_df = pd.DataFrame(
        cosine_sim, index=data["destination_name"], columns=data["destination_name"]
    )

    return data, cosine_sim_df


# Function to compute recommendations
def destination_recommendations(
    nama_destinasi,
    data,
    cosine_sim_df,
    k=6,
):
    similarity_data = cosine_sim_df
    items = data[
        ["destination_name", "category", "price", "description", "rating", "image"]
    ]

    index = (
        similarity_data.loc[:, nama_destinasi]
        .to_numpy()
        .argpartition(range(-1, -k, -1))
    )
    closest = similarity_data.columns[index[-1 : -(k + 2) : -1]]
    closest = closest.drop(nama_destinasi, errors="ignore")

    # Return recommendations as list of dictionaries
    recommendations = []
    for dest_name in closest:
        dest_row = items[items["destination_name"] == dest_name].iloc[0]
        recommendations.append(
            {
                "destination_name": dest_row["destination_name"],
                "category": dest_row["category"],
                "price": int(dest_row["price"]),  # Convert price to int
                "description": dest_row["description"],
                "rating": int(dest_row["rating"]),  # Convert rating to int
                "image": dest_row["image"],
            }
        )

    return recommendations
