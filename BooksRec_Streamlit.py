import streamlit as st


import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
import string
from fuzzywuzzy import fuzz


dfbook = pd.read_csv('Books.csv', error_bad_lines= False)
dfbook_kaggle = pd.read_csv('Books_kaggle.csv',encoding="latin-1", error_bad_lines= False)
dfusers = pd.read_csv('Users.csv', error_bad_lines= False)
dfrating = pd.read_csv('Ratings.csv', encoding="latin-1", error_bad_lines= False)

# dfbook_kaggle
# Drop de colunas 
dfbook_kaggle.drop(['Image-URL-S', 'Image-URL-M', 'Image-URL-L'], axis = 1, inplace = True)
dfbook_kaggle.rename(columns = {'Book-Title':'title', 'Book-Author':'author', 'Year-Of-Publication':'year', 'Publisher':'publisher'}, inplace=True)

# Renomeando as colunas dfrating
dfrating.rename(columns = {'User-ID':'user_id', 'Book-Rating':'rating'}, inplace=True)

dfmerge = dfbook_kaggle.merge(dfrating, on = 'ISBN', how = 'inner')

dfmerge_filtrada = dfmerge[dfmerge['rating']>0]
dfmerge_filtrada.drop_duplicates(['user_id','title'], inplace=True)

df_aval_user = dfmerge_filtrada.groupby('user_id')['title'].count().sort_values(ascending=False).reset_index()
limites = [0, 2, 50, 100, 1000, 10000]
df_aval_user['num_aval_user_segmentado'] = pd.cut(df_aval_user['title'], bins = limites)
df_aval_user.groupby('num_aval_user_segmentado')['user_id'].count().reset_index()
df_aval_user.rename(columns= {'title':'number_aval_user'}, inplace=True)
# Juntando as bases...
dfmerge_filtrada = dfmerge_filtrada.merge(df_aval_user, on = 'user_id')
dfmerge_filtrada = dfmerge_filtrada[dfmerge_filtrada['number_aval_user']<1000]
dfmerge_filtrada.drop('num_aval_user_segmentado', axis=1, inplace=True)


number_rating = dfmerge_filtrada.groupby('title')['rating'].count().reset_index()
number_rating.rename(columns= {'rating':'number_of_ratings'}, inplace=True)
dfmerge_filtrada = dfmerge_filtrada.merge(number_rating, on = 'title')
dfmerge_filtrada = dfmerge_filtrada[dfmerge_filtrada['number_of_ratings']>20]


dffinal = dfmerge_filtrada

dfbook_pivot = dffinal.pivot_table(columns = 'user_id', index = 'title', values = 'rating').fillna(0)
dfbook_sparse = csr_matrix(dfbook_pivot)

model_knn = NearestNeighbors(metric = 'cosine', algorithm='brute')
model_knn.fit(dfbook_sparse)

def print_book_recommendations(query_book, book_review_matrix, knn_model, k):
    """
    Inputs:
    query_book: nome do livro o qual queremos recomendações
    book_review_matrix: dataframe com o review count dataframe (o do pandas dataframe, não a matriz esparsa)
    knn_model: modelo que treinamos
    k: quantidade de vizinhos
    """
    # inicialização de variáveis
    query_index = None
    ratio_tuples = []
    
    
    for i in book_review_matrix.index:
        # faz a busca 'fuzzy' - adiciona se for parecido com a query que foi informada na entrada da função
        ratio = fuzz.ratio(i.lower(), query_book.lower())
        if ratio >= 75:
            current_query_index = book_review_matrix.index.tolist().index(i)
            ratio_tuples.append((i, ratio, current_query_index))
   
    # apresenta resultados
    print('Possible matches: {0}\n'.format([(x[0], x[1]) for x in ratio_tuples]))
    
    # captura o índice do livro teve o melhor match 
    try:
        query_index = max(ratio_tuples, key = lambda x: x[1])[2] # get the index of the best artist match in the data
    except:
        print('Your book didn\'t match any book in the data. Try again')
        return None
    
    # formatação da entrada do modelo e chamada
    vetor = np.array(book_review_matrix.iloc[query_index, :])
    distances, indices = knn_model.kneighbors(vetor.reshape(1, -1), n_neighbors = k)

    # apresenta os livros selecionados 
    for i in range(0, len(distances.flatten())):
        if i == 0:
            st.subheader('Recommendations for {0}:\n'.format(book_review_matrix.index[query_index]))
        else:
            st.success('{0}: {1}, with distance of {2}:'.format(i, book_review_matrix.index[indices.flatten()[i]], distances.flatten()[i]))

    return None



# Text/Title
st.title("Recomendação de Livros")

# Capturar input do usuário e fazer a consulta
nome = st.text_input("Digite o nome do Livro","Digite aqui...")

if st.button("Submeter"):
    data = dfbook_pivot
    modelo = model_knn
    print_book_recommendations(nome, data, modelo, k = 11)