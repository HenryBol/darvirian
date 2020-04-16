# Kaggle challenge Covid-19
# Team: Darvirian
# Developer: Henry Bol

# Contents:
# PART I: Load the data
# PART II: Preprocessing
# PART III: Tokenize in sentences and words
# PART IV: Vectorize (and calculate TF-IDF)
# PART V: Create the worddic with per word: doc, positions in doc, TF-IDF score

# Credits:
# Inspiration: https://www.kaggle.com/amitkumarjaiswal/nlp-search-engine
# CSV files from: https://www.kaggle.com/xhlulu/cord-19-eda-parse-json-and-generate-clean-csv


# =============================================================================
# Import the libraries
# =============================================================================
import re
import pickle
import pandas as pd
# import numpy as np

from collections import Counter
# from collections import OrderedDict
from collections import defaultdict

from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords

#from nltk.stem.porter import PorterStemmer
#from nltk.stem import SnowballStemmer

from sklearn.feature_extraction.text import TfidfVectorizer


# =============================================================================
# PART I: Load the data
# =============================================================================
## Read docs from CORD-19
# import os
# os.chdir("../Data/CSV")
df_biorxiv = pd.read_csv('Data/CSV/biorxiv_clean.csv')
df_clean_comm_use = pd.read_csv('Data/CSV/clean_comm_use.csv')
df_clean_noncomm_use = pd.read_csv('Data/CSV/clean_noncomm_use.csv')
df_clean_pmc = pd.read_csv('Data/CSV/clean_pmc.csv')

# Add all dataframes togethers
df = df_biorxiv.append(df_clean_comm_use).reset_index(drop=True)
df = df.append(df_clean_noncomm_use).reset_index(drop=True)
df = df.append(df_clean_pmc).reset_index(drop=True)

# Select dataset (test purposes)
# df = df_biorxiv.copy()


## Series plot_data with text (all documents)
plot_data = df['text']

# TODO check documents in other languages than English (e.g. German)

# Create Documentnumber to PaperID table
# doc_to_paper_id = df.paper_id.reset_index()
# doc_to_paper_id.to_csv('Data/output/doc_to_paper.csv')

# Slice for short df
# df.columns
# df = df[['paper_id', 'title', 'authors', 'affiliations', 'abstract', 'bibliography']]
# # df.to_csv('Data/output/df.csv')
# # df = pd.read_csv('Data/output/df.csv')
# f = open("Data/output/df.pkl","wb")
# pickle.dump(df, f)
# f.close()


# =============================================================================
# PART II: Preprocessing
# =============================================================================
# Check NaNs
# df.isnull().values.any()
# df.isna().any() # title, authors, afffiliations, avstract
# NaN_list_rows = df.isnull().sum(axis=1).sort_values(ascending=False)
# df = df.replace(np.nan, '', regex=True)
# plot_data.isnull().values.any() # False


## Check duplicates
duplicate_papers = df[df.paper_id.duplicated()] # None


## ! Check without removing the '\n' Keep original sentences and tokenize
sentences = [sent_tokenize(plot_data[i]) for i in range(len(plot_data)) if len(plot_data[i]) != 0]


## Save sentences file
f = open("Data/output/sentences_200415.pkl","wb")
pickle.dump(sentences, f)
f.close()

# Load pickle file sentences
# if inference == 'on':
#     pickle_in = open("Data/output/sentences_200415.pkl", "rb")
#     sentences = pickle.load(pickle_in)


## Replace '\n' by ' '
plot_data = [x.replace('\n', ' ') for x in plot_data]

# TODO include '-' 
# rank('Full-genome phylogenetic analysis'): full-genome is not taken into account
# rank('Full genome phylogenetic analysis'): full genome is taken into account


## Clean text
# Keep characters only
# plot_data = [re.sub(r'[^a-zA-Z. ]', '', str(x)) for x in plot_data]
plot_data = [re.sub(r'[^a-zA-Z]', ' ', str(x)) for x in plot_data]

# Remove single characters
plot_data = [re.sub(r'\b[a-zA-Z]\b', '', str(x)) for x in plot_data]

# Remove punctuation (!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~)
# plot_data = ["".join(j for j in i if j not in string.punctuation) for i in plot_data]


# =============================================================================
# PART III: Tokenize and preprocess more
# =============================================================================
## Tokenize words
plot_data = [word_tokenize(doc) for doc in plot_data]


## Lower case words for all docs
plot_data = [[word.lower() for word in doc] for doc in plot_data]


## Remove stop words from all docs
stop_words = set(stopwords.words('english'))
plot_data = [[word for word in doc if word not in stop_words] for doc in plot_data]


## Remove words that occur only once in all documents
# Check frequency of words and sort from high to low
num_of_words = Counter(word for doc in plot_data for word in set(doc))
# num_of_words_sorted = OrderedDict(num_of_words.most_common())
num_of_words_sorted = [(l,k) for k,l in sorted([(j,i) for i,j in num_of_words.items()], reverse=True)]

# All words with a frequency of 1 (word[0] is a word and word[1] the frequency)
words_low_freq = [word[0] for word in num_of_words_sorted if word[1] == 1]
# Set to increase speed
words_low_freq = set(words_low_freq)

# Remove words with a frequency of 1 (this takes a while) = this takes too much time
# plot_data = [[word for word in doc if word not in words_low_freq] for doc in plot_data]
plot_data = [[word for word in doc if word not in words_low_freq] for doc in plot_data]

## TODO Stemming or lemmatization
#snowball_stemmer = SnowballStemmer("english")
#stemmed_sentence = [snowball_stemmer.stem(w) for w in filtered_sentence]
#stemmed_sentence[0:10]
#
#porter_stemmer = PorterStemmer()
#snowball_stemmer = SnowballStemmer("english")
#stemmed_sentence = [porter_stemmer.stem(w) for w in filtered_sentence]
#stemmed_sentence[0:10]


## Save plot_data file
f = open("Data/output/plot_data_200415-2.pkl", "wb")
pickle.dump(plot_data, f)
f.close()

## Load pickle file plot_data
# if inference == 'on':
#     pickle_in = open("Data/output/plot_data_200415.pkl", "rb")
#     plot_data = pickle.load(pickle_in)


# =============================================================================
# PART IV: Vectorize (and calculate TF-IDF)
# ============================================================================
texts_flattened = [" ".join(x) for x in plot_data]
# vectorizer = TfidfVectorizer(lowercase=True, analyzer='word', stop_words='english')

# Include with token_pattern also single characters
vectorizer = TfidfVectorizer(lowercase=False, stop_words=None, token_pattern=r"(?u)\b\w+\b")
vectors = vectorizer.fit_transform(texts_flattened)
feature_names = vectorizer.get_feature_names()
dense = vectors.todense()


## Dictionary of unique words as values
word2idx = dict(zip(feature_names, range(len(feature_names))))

# Save word2idx file
f = open("Data/output/word2idx_200415-2.pkl", "wb")
pickle.dump(word2idx, f)
f.close()

# Dictionary with the unique words as keys
idx2word = {v:k for k,v in word2idx.items()}

## Save idx2word file
f = open("Data/output/idx2word_200415-2.pkl", "wb")
pickle.dump(idx2word, f)
f.close()

# Load pickle file idx2word
# if inference == 'on':
#     pickle_in = open("Data/output/idx2word_200415.pkl", "rb")
#     idx2word = pickle.load(pickle_in)


## word2idx all feature_names 
features_names_num = [word2idx[feature] for feature in feature_names]


## Calculate tfidf
# df_tfidf = pd.DataFrame(dense, columns=feature_names)
df_tfidf = pd.DataFrame(dense, columns=features_names_num)


## word2idx for all words in plot_data
plot_data = [[word2idx.get(word) for word in line] for line in plot_data]

# Save plot_data_num file
f = open("Data/output/plot_data_200415_num-2.pkl", "wb")
pickle.dump(plot_data, f)
f.close()

## Load pickle file plot_data_num
# if inference == 'on':
#     pickle_in = open("Data/output/plot_data_200415_num.pkl", "rb")
#     plot_data = pickle.load(pickle_in)


# =============================================================================
# PART V: Create the worddic with per word: doc, positions in doc, TF-IDF score
# =============================================================================
# Output: dictionary worddic
# KEY: word
# VALUES: list of doc indexes where the word occurs plus per doc: word position(s) and tfidf-score

# Create dictionary with a list as values
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
# Set to speed up
# doc2 = set(doc) 
import time
start = time.time()
[worddic[word].append([plot_data.index(doc), 
                        [index for index, w in enumerate(doc) if w == word], 
                        df_tfidf.loc[i, word]]) 
                        for i,doc in enumerate(plot_data) for word in set(doc)]
                        # for i,doc in enumerate(plot_data) for word in doc2]
end = time.time()
print(end - start) # duration 63 sec for biorxiv; duration 11,779 sec (3.1 hours) for all datasets


## Save pickle file
f = open("Data/output/worddic_all_200415_num-2.pkl","wb")
pickle.dump(worddic,f)
f.close()

## Load pickle file worddic
# if inference == 'on':
#     pickle_in = open("Data/output/worddic_all_200415.pkl", "rb")
#     worddic = pickle.load(pickle_in)




