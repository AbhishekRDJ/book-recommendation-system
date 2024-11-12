import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from PIL import Image, UnidentifiedImageError
import requests
from io import BytesIO
import pickle

# Load the similarity matrix and DataFrame
similarity_matrix = np.load('model/book_recommendation_similarity_matrix.npz')['arr_0']
with open('model/books_df.pkl', 'rb') as file:
    df = pickle.load(file)

# Streamlit UI
st.title("Book Recommendation System")

# Input fields for book details
author_input = st.text_input("Book Author")
year_input = st.text_input("Year of Publication")
publisher_input = st.text_input("Publisher")

if st.button("Get Recommendations"):
    # Filter the DataFrame based on input criteria
    filtered_df = df[(df['Book-Author'].str.contains(author_input, case=False, na=False)) &
                     (df['Year-Of-Publication'].astype(str) == year_input) &
                     (df['Publisher'].str.contains(publisher_input, case=False, na=False))]

    if filtered_df.empty:
        st.write("No matching books found.")
    else:
        # Get the first book index from the filtered DataFrame
        book_idx = filtered_df.index[0]
        
        # Calculate similarity scores for the selected book
        similarity_scores = list(enumerate(similarity_matrix[book_idx]))
        sorted_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)

        # Show top 5 recommended books
        st.subheader("Recommended Books:")
        for i in range(1, 6):  # Skip the first one because it's the same book
            idx = sorted_scores[i][0]
            recommended_book = df.iloc[idx]
            
            # Display book details
            st.write(f"**Title**: {recommended_book['Book-Title']}")
            st.write(f"**Author**: {recommended_book['Book-Author']}")
            st.write(f"**Publisher**: {recommended_book['Publisher']}")
            st.write(f"**Year of Publication**: {recommended_book['Year-Of-Publication']}")

            # Attempt to load the book image
            image_url = recommended_book['Image-URL-L']
            if image_url:
                try:
                    response = requests.get(image_url, timeout=5)
                    # Check if the response is an image
                    if "image" in response.headers["Content-Type"]:
                        img = Image.open(BytesIO(response.content))
                        st.image(img, width=150)
                    else:
                        st.write("Image not available (invalid image type).")
                except (requests.exceptions.RequestException, UnidentifiedImageError):
                    st.write("Image not available.")
            else:
                st.write("Image URL is missing.")
