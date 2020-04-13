# Kaggle challenge Covid-19
# Team: Darvirian
# Date: April 2nd, 2020

# Contents:
# PART I: Preprocessing
# PART II Tokenize in sentences and words
# PART III: Vectorize and calculate TF-IDF
# PART IV: Creating the inverse-index: creates 'worddic' (dict with all words: vectorized and in which document(s) it occurs on which position(s) and tfidf)
# PART V: The Search Engine: function 'search'
# PART VI: Rank and return (rules based): function 'rank' based on 5 rules and providing summaries
# CASE 0: Sustainable risk reduction strategies

# Inspiration: https://www.kaggle.com/amitkumarjaiswal/nlp-search-engine

# TODO
# limit number of words via
# max_vocab_size
# stemming
# preference: lemmatization


# =============================================================================
# Initialization
# =============================================================================
# Train or Inference
inference = 'on' # 'on' or 'off'

# Select dataset
# dataset = 'biorxiv'
dataset = 'all'


# =============================================================================
# Import the libraries
# =============================================================================
## Importing the libraries
import re
import pickle
import pandas as pd
import numpy as np
import string

from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords

#from nltk.stem.porter import PorterStemmer
#from nltk.stem import SnowballStemmer

from sklearn.feature_extraction.text import TfidfVectorizer


# =============================================================================
# LOAD THE DATA
# =============================================================================
## Read docs from CORD-19
import os
# os.chdir("../Data/CSV")
df_biorxiv = pd.read_csv('Data/CSV/biorxiv_clean.csv')
df_clean_comm_use = pd.read_csv('Data/CSV/clean_comm_use.csv')
df_clean_noncomm_use = pd.read_csv('Data/CSV/clean_noncomm_use.csv')
df_clean_pmc = pd.read_csv('Data/CSV/clean_pmc.csv')

# Add all dataframes togethers
if dataset == 'all':
    df = df_biorxiv.append(df_clean_comm_use).reset_index(drop=True)
    df = df.append(df_clean_noncomm_use).reset_index(drop=True)
    df = df.append(df_clean_pmc).reset_index(drop=True)

# Select dataset
if dataset == 'biorxiv':
    df = df_biorxiv.copy()


## Copy for convenience the text data to a 'Raw_Text' column
df['Raw_Text'] = df['text']

# Create Documentnumber to PaperID table
# doc_to_paper_id = df.paper_id.reset_index()
# doc_to_paper_id.to_csv('Data/output/doc_to_paper.csv')

# Slice for short df
df.columns
df = df[['paper_id', 'title', 'authors', 'affiliations', 'abstract', 'bibliography']]
# df.to_csv('Data/output/df.csv')
# df = pd.read_csv('Data/output/df.csv')
f = open("Data/output/df.pkl","wb")
pickle.dump(df, f)
f.close()

# =============================================================================
# PART I: PREPROCESSING
# =============================================================================
# Keep only a few docs to speed up processing for development purposes
# df = df[:3]

# Check NaNs
df.isnull().values.any()
df.isna().any() # title, authors, afffiliations, avstract
NaN_list_rows = df.isnull().sum(axis=1).sort_values(ascending=False)
df = df.replace(np.nan, '', regex=True)

## Check duplicates
duplicate_papers = df[df.paper_id.duplicated()] # None

# Replace '\n' by ' '
df['Raw_Text'] = [x.replace('\n', ' ') for x in df['Raw_Text']]

## Keep orginal sentences and tokenize (and store in df.Sentences)
df['Sentences'] = None
df.Sentences = [sent_tokenize(df.Raw_Text[i]) for i in range(len(df)) if len(df.Raw_Text[i]) != 0]

# TODO Check "" and '' quotes in sentences

sentences = df.Sentences

## Save sentences file
# f = open("Data/output/sentences_200410.pkl","wb")
# pickle.dump(sentences, f)
# f.close()

## Load pickle file sentences
if inference == 'on':
    pickle_in = open("Data/output/sentences_200410.pkl", "rb")
    sentences = pickle.load(pickle_in)


## Clean text (keep '.' for tokenize sentences); check add characters e.g. '-' add or not (also used as hyphen)
# df.Raw_Text = [re.sub(r'[^a-zA-Z0-9. ]', '', str(x)) for x in df.Raw_Text]
# df.Raw_Text = [re.sub(r'[^a-zA-Z.\- ]', '', str(x)) for x in df.Raw_Text]
# take also '-' out
df.Raw_Text = [re.sub(r'[^a-zA-Z. ]', '', str(x)) for x in df.Raw_Text]
# TODO check Replace'-' by space ' '
# df.Raw_Text = [re.sub(r'[-]', ' ', str(x)) for x in df.Raw_Text]


# =============================================================================
# PART II: TOKENIZE IN SENTENCES AND WORDS
# =============================================================================
# Remove punctuation from all documents and create alldocslist
alldocslist = list(df.Raw_Text)


## Remove punctuation
alldocslist = ["".join(j for j in i if j not in string.punctuation) for i in alldocslist]


## Tokenize words (and store in plot_data)
plot_data = [word_tokenize(doc) for doc in alldocslist]


## Lower case words for all docs
plot_data = [[w.lower() for w in line] for line in plot_data]


## Remove stop words from all docs
stop_words = set(stopwords.words('english'))
plot_data = [[w for w in line if w not in stop_words] for line in plot_data]


## TODO Stemming or lemmatization
## Stem words EXAMPLE (could try others/lemmers) / these stemmers are not so good; is not used
#snowball_stemmer = SnowballStemmer("english")
#stemmed_sentence = [snowball_stemmer.stem(w) for w in filtered_sentence]
#stemmed_sentence[0:10]
#
#porter_stemmer = PorterStemmer()
#snowball_stemmer = SnowballStemmer("english")
#stemmed_sentence = [porter_stemmer.stem(w) for w in filtered_sentence]
#stemmed_sentence[0:10]


## Save plot_data file
f = open("Data/output/plot_data_200411.pkl", "wb")
pickle.dump(plot_data, f)
f.close()

## Load pickle file plot_data
if inference == 'on':
    pickle_in = open("Data/output/plot_data_200411.pkl", "rb")
    plot_data = pickle.load(pickle_in)


# =============================================================================
# PART III: VECTORIZE (and calculate TF-IDF)
# ============================================================================
texts_flattened = [" ".join(x) for x in plot_data]
# vectorizer = TfidfVectorizer(lowercase=True, analyzer='word', stop_words='english')

# Include with token_pattern also single characters
vectorizer = TfidfVectorizer(lowercase=False, stop_words=None, token_pattern=r"(?u)\b\w+\b")
vectors = vectorizer.fit_transform(texts_flattened)
feature_names = vectorizer.get_feature_names()
dense = vectors.todense()

df_tfidf = pd.DataFrame(dense, columns=feature_names)

# Get word -> integer mapping
# word2idx = tokenizer.word_index
# V = len(word2idx)
# print('Found %s unique tokens.' % V)


## Save pickle file (too big to create)
# f = open("Data/output/df_tfidf_200410.pkl", "wb")
# pickle.dump(df_tfidf, f)
# f.close()

## Load pickle file df_dfidf
# if inference == 'on':
#     pickle_in = open("Data/output/df_tfidf_200410.pkl","rb")
#     df_tfidf = pickle.load(pickle_in)


# =============================================================================
# PART IV: CREATING THE INVERSE-INDEX
# =============================================================================
# Create inverse index which gives document number for each document and where word appears

## Check unique words
# all_words = [item for sublist in plot_data for item in sublist]
# wordsunique = set(all_words)
# wordsunique = list(wordsunique)
# len(wordsunique)


# ## Dictionary of unique words as values
# idx2word = dict(enumerate(wordsunique))
# # Dictionary with the unique words as keys
# word2idx = {v:k for k,v in idx2word.items()}


# # Back-up copy
# plot_data_copy = plot_data.copy()

# # words2idx on plot_data
# plot_data = [[word2idx.get(w) for w in doc] for doc in plot_data]


## Create dictonary of words
# Output: dictionary worddic
# KEY: word
# VALUES: list of doc indexes where the word occurs plus per doc: word position(s) and tfidf-score

# Train TF-IDF if inference is off (no on)
if inference != 'on':

    # Create dictionary with a list as values
    from collections import defaultdict
    worddic = defaultdict(list)

    # Loop (for reference and to make the comprehension (see below) a bit more understandable)
    # for i,doc in enumerate(plot_data):
    #     for doc in plot_data:
    #         for word in set(doc): # set provides unique words in doc
    #             print(word)
    #             # word = str(word)
    #             index = plot_data.index(doc)
    #             positions = [index for index, w in enumerate(doc) if w == word]
    #             idfs = df_tfidf.loc[i, word]
    #             worddic[word].append([index,positions,idfs])

    # Create the dictionary via comprehension to speed up processing
    import time
    start = time.time()
    [worddic[word].append([plot_data.index(doc), 
                            [index for index, w in enumerate(doc) if w == word], 
                            df_tfidf.loc[i, word]]) 
                            for i,doc in enumerate(plot_data) for word in set(doc)]
 
    end = time.time()
    print(end - start) # duration 76 sec for biorxiv; duration 11314 sec (3.142 hours) for all datasets

    ## Save pickle file
    f = open("Data/output/worddic_all_200410.pkl","wb")
    pickle.dump(worddic,f)
    f.close()

## Load pickle file worddic
# if inference == 'on':
#     pickle_in = open("Data/output/worddic_all_200410.pkl", "rb")
#     worddic = pickle.load(pickle_in)


## Split dictionary into keys and values
# keys = worddic.keys()
# values = worddic.values()
# items = worddic.items()
# worddic_list = list(worddic.items())

