import streamlit as st
import pandas as pd
from scipy.sparse import load_npz

# Load the sparse similarity matrix and DataFrame
similarity_matrix = load_npz('model/book_recommendation_similarity_matrix.npz')
df = pd.read_pickle('model/books_df.pkl')

# Function to get recommendations
def get_recommendations(book_author, year_of_publication, publisher, df, similarity_matrix):
    input_combined = f"{book_author} {publisher} {year_of_publication}"
    
    # Find the closest match for the input
    if input_combined in df['combined_features'].values:
        book_idx = df[df['combined_features'] == input_combined].index[0]
    else:
        st.write('---')
        st.write("No exact match found. Showing similar books.")
        st.write('---')
        book_idx = 0  # Default index if no exact match found

    # Get similarity scores (only retrieve the non-zero values)
    similarity_scores = list(enumerate(similarity_matrix[book_idx].toarray().flatten()))
    similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
    
    # Get top 5 similar books
    recommendations = [df.iloc[i[0]] for i in similarity_scores[1:6]]
    return recommendations

# Streamlit UI
st.title("Book Recommendation System")
st.write("Find books similar to your preferences")

# Input fields for recommendation
author = st.text_input("Book Author")
year_of_publication = st.text_input("Year of Publication")
publisher = st.text_input("Publisher")

if st.button("Get Recommendations"):
    # Get recommendations based on the input
    recommended_books = get_recommendations(author, year_of_publication, publisher, df, similarity_matrix)
    
    for book in recommended_books:
        st.write(f"**Title**: {book['Book-Title']}")
        st.write(f"**Author**: {book['Book-Author']}")
        st.write(f"**Publisher**: {book['Publisher']}")
        st.write(f"**Year of Publication**: {book['Year-Of-Publication']}")
        
        # Add separation line between books
        st.write('---')  # This adds a divider between each book's details
