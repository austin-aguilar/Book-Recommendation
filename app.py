import pickle
import numpy as np
import pandas as pd
import streamlit as st

#In terminal type 'streamlit run app.py'
st.header("Book Recommendation System Using Machine Learning")
model = pickle.load(open("./artifacts/model.pkl","rb"))
books_title = pickle.load(open("./artifacts/book_title.pkl","rb"))
final_rating = pickle.load(open("./artifacts/popular_books.pkl","rb"))
book_pivot = pickle.load(open("./artifacts/book_pivot.pkl","rb"))
SVD_matrix = pickle.load(open("./artifacts/SVD_matrix.pkl","rb"))

def fetch_poster(suggestion):
    ids = [np.where(final_rating['title'] == i)[0][0] for i in suggestion]
    poster_url = [final_rating.iloc[i]['url'] for i in ids]
    return poster_url

# def recommend_books(book_name):
#     book_id = np.where(book_pivot.index == book_name)[0][0]
#     distance, suggestion = model.kneighbors(book_pivot.iloc[book_id,:].values.reshape(1,-1),n_neighbors=6)
#     poster_url = fetch_poster(suggestion)
#     results=[]
#     for i in range(len(suggestion)):
#         books = book_pivot.index[suggestion[i]]
#         for j in books:
#             results.append(j)
#     return results, poster_url

def recommend_books_SVD (book_title):
    corr = np.corrcoef(SVD_matrix)
    books = book_pivot.T
    books_cols = book_pivot.T.columns
    books_list=list(books_cols)
    book = books_list.index(book_title)
    corr_book = corr[book] 
    # Create a DataFrame with book titles and scores
    df = pd.DataFrame({'title': books_cols, 'score': corr_book})
    # Filter for scores between 0.75 and 1
    similar = df[(df['score'] < 1.0) & (df['score'] > 0.75)]
    similar = similar.sort_values(by='score', ascending=False)
    results = list(similar['title'])
    poster_url = fetch_poster(results)
    return results, poster_url



selected_books = st.selectbox(
    "Type or select a book",
    books_title
)

if st.button('Show Recommendation'):
    recommendations_books, poster_url = recommend_books_SVD(selected_books)
    col1, col2, col3, col4, col5 = st.columns(5)
    col_list = [col1,col2,col3,col4,col5]
    for i in range(len(col_list)):
        with col_list[i]:
            st.text(recommendations_books[i+1])
            st.image(poster_url[i+1])
   
   
    # with col1:
    #     st.text(recommendations_books[1])
    #     st.image(poster_url[1])