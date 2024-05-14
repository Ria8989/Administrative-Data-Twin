import sys

# Specify the custom folder path containing the Python modules
custom_folder_path = '/home/adt/.local/lib/python3.10/site-packages/'

# Add the custom folder path to the sys.path list
sys.path.append(custom_folder_path)

# %%
import pandas as pd
import numpy as np
import re, os
import ast
import nltk
#nltk.download()
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from collections import OrderedDict
from collections import Counter

import pickle

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import LatentDirichletAllocation

import gensim
from gensim.models.word2vec import Word2Vec
from gensim.models import KeyedVectors
from sklearn.metrics.pairwise import cosine_similarity
from gensim import corpora, models
from gensim.models import LdaModel


# %%
# Define the relative paths
relative_path_vec = "./glove_wiki_vectors.kv"
relative_path_embedding = "./oov_word_embeddings.pkl"
relative_path_sdg = "./sdg_key.csv"
relative_path_input = "./input.txt"
relative_path_output = "./output.csv"

# Get the directory of the current Python file being executed if available, else use the current working directory
current_dir = os.path.dirname(os.path.abspath(__file__)) if "__file__" in locals() else os.getcwd()

# Get the absolute paths by joining the current directory with the relative paths
absolute_path_vec = os.path.normpath(os.path.join(current_dir, relative_path_vec))
absolute_path_embedding = os.path.normpath(os.path.join(current_dir, relative_path_embedding))
absolute_path_sdg = os.path.normpath(os.path.join(current_dir, relative_path_sdg))
absolute_path_input = os.path.normpath(os.path.join(current_dir, relative_path_input))
absolute_path_output = os.path.normpath(os.path.join(current_dir, relative_path_output))


# %%
# Load the saved word embeddings dictionary from the file
with open(absolute_path_embedding, 'rb') as f:
    oov_word_embeddings = pickle.load(f)

# %%
wv = KeyedVectors.load(absolute_path_vec)

# %%
missing_word = []

# %%
def sent_vec_avg(sent):
    vector_size = wv.vector_size
    missing_words = []
    vectors = []
    for w in sent:
        ctr = 0
        wv_res = np.zeros(vector_size)
        for word in  w:
            if word in wv:
                wv_res += wv[word]
                ctr += 1
            elif word in oov_word_embeddings:
                wv_res += oov_word_embeddings[word]
                ctr += 1
            else:
                missing_word.append(w)
        if ctr>0:
            wv_res = wv_res/ctr
        vectors.append(wv_res)
    vec_arr = np.array(vectors)
    len_vec = len(vectors)
    return vec_arr

# %%
sdg = pd.read_csv(absolute_path_sdg, sep=";", dtype={'Id': str})

# %%
sdg['new_description'] = sdg['new_description'].apply(ast.literal_eval)
sdg['keywords'] = sdg['keywords'].apply(ast.literal_eval)

# %%
sdg['vec'] = sdg['keywords'].apply(sent_vec_avg)

# %%
description_map = dict(zip(sdg['Id'], sdg['Description']))

# %%


# %%
# Step 1: Read the document
def read_document(file_path):
    with open(file_path, 'r') as file:
        document = file.read()
        document=document.lower()
    return document

# %%
# Step 2: Tokenize
def tokenize(document):
    tokens = word_tokenize(document)
    # Filter out tokens that are not alphanumeric
    tokens = [token for token in tokens if token.isalpha()]
    return tokens

# %%
# Step 3: Lemmatize
def lemmatize(tokens):
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]
    return lemmatized_tokens

# %%
# Step 4: Remove stopwords
def remove_stopwords(tokens):
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [token for token in tokens if token.lower() not in stop_words]
    return filtered_tokens

# %%
# Step 5: Get top k words
def get_top_k_words(tokens, k=10):
    word_freq = Counter(tokens)
    top_k_words = [word for word, _ in word_freq.most_common(k)]
    return top_k_words

# %%
def get_top_p_percent_words(tokens, p):
    # Calculate the value of k as p percent of the total number of tokens
    k = max(int(len(set(tokens)) * p), 1)  # Ensure k is at least 1
    word_freq = Counter(tokens)
    top_k_words = [word for word, _ in word_freq.most_common(k)]
    return top_k_words

# %%
# Step 6: Topic Modeling
def topic_modeling(tokens, n_topics):
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(tokens)
    lda_model = LatentDirichletAllocation(n_components=n_topics, random_state=42)
    lda_model.fit(X)
    return lda_model, vectorizer

# %%
# Step 7: Choose top n topics
def choose_top_topics(lda_model, vectorizer, n_topics, n_words=10):
    topics = lda_model.components_
    topic_keywords = []
    for topic_idx, topic in enumerate(topics):
        top_n_indices = topic.argsort()[-n_words:][::-1]
        topic_keywords.append([vectorizer.get_feature_names_out()[i] for i in top_n_indices])
    return topic_keywords

# %%
# Step 8: Extract keywords from each topic
def extract_keywords_from_topics(topic_keywords):
    key_words = []
    for item in topic_keywords:
        key_words.extend(item)
    return key_words

# %%
# Step 9: Merge set 1 and set 2
def merge_sets(list1, list2):
    merged_set = list1 + list2 
    return list(set(merged_set))

# %%
# Main function to orchestrate the steps
def document_keywords_extraction(file_path, p_percent=0.2, n_topics=3):
    
    document = read_document(file_path)
    tokens = tokenize(document)
    tokens = lemmatize(tokens)
    tokens = remove_stopwords(tokens)
    top_k_words = get_top_p_percent_words(tokens, p_percent)
    lda_model, vectorizer = topic_modeling(tokens, n_topics)
    top_topics = choose_top_topics(lda_model, vectorizer, n_topics)
    keywords_from_topics = extract_keywords_from_topics(top_topics)
    merged_set = merge_sets(top_k_words, keywords_from_topics)
    
    return merged_set


# %%
doc_keywords = document_keywords_extraction(absolute_path_input)

# %%


# %%


# %%
"""
String Matching
"""

# %%


# %%
def count_common_words(row, input_list):
    set_1 = set(row)
    set_2 = set(input_list)
    common_words = len(set_1.intersection(set_2))
    words_list = list(set_1.intersection(set_2))
    return common_words, words_list

# %%
sdg_new = sdg.copy()

# %%
common_word_df = pd.DataFrame(sdg_new[['Id', 'Description']], columns = ['Id', 'Description', 'comm_word_count', 'comm_words'])

# %%
common_word_df[['comm_word_count', 'comm_words']] = sdg_new['new_description'].apply(lambda x: pd.Series(count_common_words(x, doc_keywords)))

# %%


# %%


# %%
"""
Semantic Mapping
"""

# %%
sgd_sem = sdg.copy()

# %%
def sent_vec(sent):
    vector_size = wv.vector_size
    vectors = []
    miss_w = []
    for w in sent:
        wv_res = np.zeros(vector_size)
        if w in wv:
            wv_res = wv[w]
            vectors.append(wv_res)
        elif w in oov_word_embeddings:
            wv_res = oov_word_embeddings[w]
            vectors.append(wv_res)
        else:
            missing_word.append(w)
    vec_arr = np.array(vectors)
    len_vec = len(vectors)
    return vec_arr

# %%
def find_similarity_semantic(sdg_df, doc_df, threshold):
    ind_list = sdg_df['vec'].tolist()
    id_list = sdg_df['Id'].tolist()
    similarity_values = {}
    values = {}
    for ind in range(len(ind_list)):
        max_list = []
        for i in ind_list[ind]:
            
            similarity_matrix = cosine_similarity([i], doc_df)
            max_similarity = np.max(similarity_matrix)
            max_list.append(max_similarity)
        similarity_values[id_list[ind]] = np.mean(max_list)
        values[id_list[ind]] = max_list
    return similarity_values, values


# %%
doc_vec = sent_vec(doc_keywords)

# %%
similarity_score, average_values = find_similarity_semantic(sgd_sem, doc_vec, 10)

# %%
sem_score = common_word_df.copy()

# %%
sem_score['semantic_score'] = sem_score['Id'].map(similarity_score)
sem_score['semantic_values'] = sem_score['Id'].map(average_values)

# %%


# %%
sdg_scoring = sem_score.copy()

# %%
# Assuming df is your DataFrame and 'column_name' is the column you want to normalize
scaler = MinMaxScaler()
sdg_scoring['norm_word_count'] = scaler.fit_transform(sdg_scoring[['comm_word_count']])

# %%
def scoring(row):
    word_s = row['norm_word_count'] * 0.5
    sem_s = row['semantic_score'] * 0.5
    score = (word_s + sem_s) * 100
    rounded_score = round(score, 2)  # Round to two decimal places
    return rounded_score

# %%
sdg_scoring['final_score_%'] = sdg_scoring.apply(scoring, axis=1)
#sdg_scoring

# %%
sorted_sdg_scoring = sdg_scoring.sort_values(by='final_score_%', ascending=False)
#sorted_sdg_scoring

# %%
goals = sdg_scoring[sdg_scoring['Id'].str.count('\.')==0].copy()
# Sort the DataFrame in descending order of 'col1'
sorted_goals = goals.sort_values(by='final_score_%', ascending=False)
sorted_goals.rename(columns= {'Id': 'Goal_Id', 'Description': 'Goal_Description', 'final_score_%': 'Goal_mapping_score_%'}, inplace=True)
sorted_goals.reset_index(inplace=True, drop=True)
#sorted_goals

# %%
targets = sdg_scoring[sdg_scoring['Id'].str.count('\.')==1].copy()
sorted_targets = targets.sort_values(by='final_score_%', ascending=False)
sorted_targets.rename(columns= {'Id': 'Target_Id', 'Description': 'Target_Description', 'final_score_%': 'Target_mapping_score_%'}, inplace=True)
sorted_targets.reset_index(inplace=True, drop=True)
#sorted_targets

# %%
indicators = sdg_scoring[sdg_scoring['Id'].str.count('\.')==2].copy()
sorted_indicators = indicators.sort_values(by='final_score_%', ascending=False)
sorted_indicators.rename(columns= {'Id': 'Indicator_Id', 'Description': 'Indicator_Description', 'final_score_%': 'Indicator_mapping_score_%'}, inplace=True)
sorted_indicators.reset_index(inplace=True, drop=True)
#sorted_indicators

# %%
# Select the top 10 rows from each dataframe
top_10_goals = sorted_goals.head(10)[['Goal_Id', 'Goal_Description', 'Goal_mapping_score_%']]
top_10_targets = sorted_targets.head(10)[['Target_Id', 'Target_Description', 'Target_mapping_score_%']]
top_10_indicators = sorted_indicators.head(10)[['Indicator_Id', 'Indicator_Description', 'Indicator_mapping_score_%']]

# Merge the top 10 rows from each dataframe
sdg_level = pd.concat([top_10_goals, top_10_targets, top_10_indicators], axis=1)
sdg_level = sdg_level.fillna(np.nan)
#sdg_level

# %%
sdg_level.to_csv(absolute_path_output, index=False)
