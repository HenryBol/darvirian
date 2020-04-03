# Kaggle challenge Covid-19
# Team: Darvirian
# Date: April 2nd, 2020

# Contents:
# PART I: Preprocessing
# PART II Tokenize in sentences and words
# PART III: Creating the inverse-index: creates 'worddic' (dict with all words: vectorized and in which document(s) it occurs on which position(s) and tfidf)
# PART IV: The Search Engine: function 'search'
# PART V: Rank and return (rules based): function 'rank' based on 5 rules and providing summaries
# CASE 0: Sustainable risk reduction strategies

# Inspiration: https://www.kaggle.com/amitkumarjaiswal/nlp-search-engine

# TODO
# Keep numerical values in Sentences



# =============================================================================
# Initialization
# =============================================================================
# Train or Inference
inference = 'on' # 'on' or 'off' 

# Select dataset
dataset = 'biorxiv' # only choice yet


# =============================================================================
# Import the libraries
# =============================================================================
## Importing the libraries
import pandas as pd
import numpy as np 
import string
import re
import pickle
import tensorflow as tf
tf.__version__

from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords

#from nltk.stem.porter import PorterStemmer
#from nltk.stem import SnowballStemmer


# =============================================================================
# LOAD THE DATA
# =============================================================================
## Read docs from CORD-19
# import os
# os.chdir("../Data/CSV")
df_biorxiv = pd.read_csv('Data/CSV/biorxiv_clean.csv')
df_clean_comm_use = pd.read_csv('Data/CSV/clean_comm_use.csv')
df_clean_noncomm_use = pd.read_csv('Data/CSV/clean_noncomm_use.csv')
df_clean_pmc = pd.read_csv('Data/CSV/clean_pmc.csv')

# Add all dataframes togethers
# df = df_biorxiv.append(df_clean_comm_use)
# df = df.append(df_clean_noncomm_use)
# df = df.append(df_clean_pmc)

# Select dataset
if dataset == 'biorxiv':
    df = df_biorxiv.copy()

# Inspect
df.info()
df.columns
## Copy for convenience the data to a 'Raw_Text' column
df['Raw_Text'] = df['text']


# =============================================================================
# PART I: PREPROCESSING
# =============================================================================
# Keep only 10 docs to speed up processing for development purposes
# df = df[:10]

# Check NaNs
df.isnull().values.any()
df.isna().any() # Raw_Text
NaN_list_rows = df.isnull().sum(axis=1).sort_values(ascending = False)
df = df.replace(np.nan, '', regex=True)

# Clean text (keep '.' for tokenize sentences); check add characters e.g. '-' add or not (also used as hyphen)
# TODO check adding figures (0-9) which increases the number of unique words significantly (and slow the tdidf process)
# df.Raw_Text = [re.sub(r'[^a-zA-Z0-9. ]', '', str(x)) for x in df.Raw_Text]
df.Raw_Text = [re.sub(r'[^a-zA-Z.\- ]', '', str(x)) for x in df.Raw_Text] 

# Remove '\n' 
df['Raw_Text'] = [x.replace('\n', '') for x in df['Raw_Text']]


# =============================================================================
# PART II: TOKENIZE IN SENTENCES AND WORDS
# =============================================================================
# Remove punctuation from all DOCs and create alldocslist
#exclude = set(string.punctuation)
alldocslist = list(df.Raw_Text)


## Tokenize sentences (and store in df.Sentences)
df['Sentences'] = None
df.Sentences = [sent_tokenize(alldocslist[i]) for i in range(len(alldocslist))]


## Remove punctuation
alldocslist = ["".join( j for j in i if j not in string.punctuation) for i in alldocslist]

#exclude = set(string.punctuation)
##alldocslist = []
#i = 0
#for i in range(len(df)):
#    text = df.Raw_Text[i]
#    text = ''.join(ch for ch in text if ch not in exclude)
#    alldocslist.append(text)
#print(alldocslist[1])


## Tokenize words (and store in plot_data)
# TODO check plotdata and structure
plot_data = [[]] * len(alldocslist)
index = 0
for doc in alldocslist:
    tokentext = word_tokenize(doc)
    plot_data[index].append(tokentext)
    index +=1
# keep only the first one (all others are the same)
plot_data = plot_data[0]
    
## Lower case words for all docs 
plot_data = [[w.lower() for w in line] for line in plot_data]

# Remove stop words from all docs 
# stop_words = set(stopwords.words('dutch'))
stop_words = set(stopwords.words('english'))
#stopwords = stopwords.union(set(['een','van'])) # add extra stopwords
plot_data = [[w for w in line if w not in stop_words] for line in plot_data]


## Stem words EXAMPLE (could try others/lemmers) / these stemmers are not so good; is not used
#snowball_stemmer = SnowballStemmer("dutch")
#stemmed_sentence = [snowball_stemmer.stem(w) for w in filtered_sentence]
#stemmed_sentence[0:10]
#
#porter_stemmer = PorterStemmer()
#snowball_stemmer = SnowballStemmer("dutch")
#stemmed_sentence = [porter_stemmer.stem(w) for w in filtered_sentence]
#stemmed_sentence[0:10]


# =============================================================================
# PART III: VECTORIZE (and trying to find a faster TF-IDF)
# =============================================================================
# from tensorflow.keras.preprocessing.text import Tokenizer
# from tensorflow.keras.preprocessing.sequence import pad_sequences

# # Tokenize
# # MAX_VOCAB_SIZE = 70000 # 20000
# # tokenizer = Tokenizer(num_words=MAX_VOCAB_SIZE, filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n', lower=True,
# #           split=' ', char_level=False, oov_token=None, document_count=0)
# tokenizer = Tokenizer(filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n', lower=True,
#           split=' ', char_level=False, oov_token=None, document_count=0)

# # Remove stopwords
# stop_words = set(stopwords.words('english'))
# alldocslist = [[w for w in doc if w not in stop_words] for doc in alldocslist]

# # Fit and transform
# tokenizer.fit_on_texts(alldocslist)
# sequences = tokenizer.texts_to_sequences(alldocslist)

# # Dictionary
# word2idx = tokenizer.word_index

# # Check value of key 'covid'
# word2idx.get('covid')

# # MAX_SEQUENCE_LENGTH
# # data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
# # data = pad_sequences(sequences)

# # Calculate tfidf
# tft.tfidf(x, vocab_size, smooth=True, name=None)


# #https://stackoverflow.com/questions/30013097/how-to-calculate-tf-idf-for-a-list-of-dict
#     # from sklearn.feature_extraction.text import TfidfVectorizer

#     # tfv = TfidfVectorizer(min_df=3,  max_features=None,
#     # strip_accents='unicode', analyzer='word',token_pattern=r'\w{1,}',
#     # ngram_range=(1,2), use_idf=1,smooth_idf=1,sublinear_tf=1,
#     # stop_words = 'english')


# # Vectorize
    

# =============================================================================
# PART III: CREATING THE INVERSE-INDEX
# =============================================================================
# Create inverse index which gives document number for each document and where word appears

## Check unique words
all_words = [item for sublist in plot_data for item in sublist]
wordsunique = set(all_words)
wordsunique = list(wordsunique)
len(wordsunique)


## Dictionary of unique words as values
idx2word = dict(enumerate(wordsunique))
# Dictionary with the unique words as keys
word2idx = {v:k for k,v in idx2word.items()}


## Functions for TD-IDF / BM25
# TODO check BM25
import math
#from textblob import TextBlob as tb
def tf(word, doc):
    return doc.count(word) / len(doc)
def n_containing(word, doclist):
    return sum(1 for doc in doclist if word in doc)
def idf(word, doclist):
    return math.log(len(doclist) / (0.01 + n_containing(word, doclist)))
def tfidf(word, doc, doclist):
    return (tf(word, doc) * idf(word, doclist))

## Create dictonary of words
# THIS ONE-TIME INDEXING IS THE MOST PROCESSOR-INTENSIVE STEP AND WILL TAKE TIME TO RUN (BUT ONLY NEEDS TO BE RUN ONCE)
# Output: dictionary worddic
# KEY: word 
# VALUES: list of doc indexes where the word occurs plus per doc: word position(s) and tfidf-score

plottest = plot_data[0:1000]

# Train TF-IDF if inference is off (no on)
if inference != 'on': 
    # words2idx on plottest
    plottest_num = []
    for doc in plottest:
        doc = [word2idx.get(w) for w in doc]
        plottest_num.append(doc)
    
    # make copies
    plottest_copy = plottest.copy()
    plottest = plottest_num.copy()
    
    # f = open("Data/output/plottest.pkl","wb")
    # pickle.dump(plottest_num,f)
    # f.close()  
    
    # Create dictionary with a list as values
    from collections import defaultdict
    worddic = defaultdict(list)
    
    # Loop (for reference and to make the comprehension (see below) a bit more understandable)
    # for doc in plottest:
    #     for word in set(doc): # set provides unique words in doc 
    #         word = str(word)
    #         index = plottest.index(doc)
    #         positions = [index for index, w in enumerate(doc) if w == word]
    #         idfs = tfidf(word,doc,plottest)
    #         worddic[word].append([index,positions,idfs])
    
    # Create the dictionary via comprehension to speed up processing
    import time
    start = time.time()
    # [worddic[word].append([plottest.index(doc), list(np.where(np.array(plottest[plottest.index(doc)]) == word)[0]), tfidf(word,doc,plottest)]) for doc in plottest for word in set(doc)]
    [worddic[word].append([plottest.index(doc), [index for index, w in enumerate(doc) if w == word], tfidf(word,doc,plottest)]) for doc in plottest for word in set(doc)]
    # TD-IDF processing direct: no impact
    # plottest_length = len(plottest)
    # [worddic[word].append([plottest.index(doc), [index for index, w in enumerate(doc) if w == word], (doc.count(word) / len(doc)) / np.log(plottest_length / sum(1 for doc in plottest if word in doc))]) for doc in plottest for word in set(doc)]
    end = time.time()
    print(end - start) # duration 2.0 hours for biorxiv
    
    
    ## Change words to string (instead of numbers)
    # TODO rewrite to comprehension
    # idx2words on worddic
    worddic_num = worddic.copy()
    word_list = list(worddic.keys())
    word_keys = [0] * len(worddic)
    for i in range(len(worddic)):
        print(i)
        word_keys[i] = idx2word.get(word_list[i])
    worddic = dict(zip(word_keys, list(worddic.values()))) 
    
    # check
    w = 133
    w = 30877
    idx2word.get(w)

    # Use plottest withs words in strings instead of numbers 
    plottest = plottest_copy.copy()


    ## Save pickle file
    f = open("Data/output/worddic_biorxiv_comm_200403.pkl","wb")
    pickle.dump(worddic,f)
    f.close()


## Load pickle file worddic
if inference == 'on':
    pickle_in = open("Data/output/worddic_biorxiv_clean_200402_2.pkl","rb")
    worddic = pickle.load(pickle_in) 

# Check
# worddic['covid']
# plottest[884][44]


## Split dictionary into keys and values 
keys = worddic.keys() 
values = worddic.values() 
items = worddic.items()
worddic_list = list(worddic.items())

# printing keys and values seperately 
# print("keys : ", str(keys)) 
# print("values : ", str(values)) 

# check first and last keys
# worddic_list[0]
# worddic_list[-1]

# Check alternative for creating worddic
# https://stackoverflow.com/questions/17366788/python-split-list-based-on-first-character-of-word
#def splitLst(x):
#    dictionary = dict()
#    for word in x: 
#       f = word[0]
#       if f in dictionary.keys():
#            dictionary[f].append(word)
#       else:
#            dictionary[f] = [word]
#    return dictionary
# e.g. splitLst(['About', 'Absolutely', 'After', 'Aint', 'Alabama', 'AlabamaBill', 'All', 'Also', 'Amos', 'And', 'Anyhow', 'Are', 'As', 'At', 'Aunt', 'Aw', 'Bedlam', 'Behind', 'Besides', 'Biblical', 'Bill', 'Billgone'])


# =============================================================================
# PART IV: The Search Engine
# =============================================================================
# create word search which takes multiple words and finds documents that contain both along with metrics for ranking:

    ## (1) Number of occurences of search words 
    ## (2) TD-IDF score for search words 
    ## (3) Percentage of search terms
    ## (4) Word ordering score 
    ## (5) Exact match bonus 

from collections import Counter

def search(searchsentence):
    try:
        # split sentence into individual words 
        searchsentence = searchsentence.lower()
        try:
            words = searchsentence.split(' ')
        except:
            words = list(words)
        enddic = {}
        idfdic = {}
        closedic = {}
        
        # remove words if not in worddic 
        realwords = []
        for word in words:
            if word in list(worddic.keys()):
                realwords.append(word)  
        words = realwords
        numwords = len(words)
        
        # make metric of number of occurences of all words in each doc & largest total IDF 
        for word in words:
            for indpos in worddic[word]:
                index = indpos[0]
                amount = len(indpos[1])
                idfscore = indpos[2]
                enddic[index] = amount
                idfdic[index] = idfscore
                fullcount_order = sorted(enddic.items(), key=lambda x:x[1], reverse=True)
                fullidf_order = sorted(idfdic.items(), key=lambda x:x[1], reverse=True)
               
        # make metric of what percentage of words appear in each doc
        combo = []
        alloptions = {k: worddic.get(k, None) for k in (words)}
        for worddex in list(alloptions.values()):
            for indexpos in worddex:
                for indexz in indexpos:
                    combo.append(indexz)
        comboindex = combo[::3]
        combocount = Counter(comboindex)
        for key in combocount:
            combocount[key] = combocount[key] / numwords
        combocount_order = sorted(combocount.items(), key=lambda x:x[1], reverse=True)
        
        # make metric for if words appear in same order as in search
        if len(words) > 1:
            x = []
            y = []
            for record in [worddic[z] for z in words]:
                for index in record:
                     x.append(index[0])
            for i in x:
                if x.count(i) > 1:
                    y.append(i)
            y = list(set(y))

            closedic = {}
            for wordbig in [worddic[x] for x in words]:
                for record in wordbig:
                    if record[0] in y:
                        index = record[0]
                        positions = record[1]
                        try:
                            closedic[index].append(positions)
                        except:
                            closedic[index] = []
                            closedic[index].append(positions)

            # TODO check fdic order, seems often to be 0: firstlist is not defined yet
            x = 0
            fdic = {}
            for index in y:
                csum = []
                for seqlist in closedic[index]:
                    while x > 0:
                        secondlist = seqlist
                        x = 0
                        sol = [1 for i in firstlist if i + 1 in secondlist]
                        csum.append(sol)
                        fsum = [item for sublist in csum for item in sublist]
                        fsum = sum(fsum)
                        fdic[index] = fsum
                        fdic_order = sorted(fdic.items(), key=lambda x:x[1], reverse=True)
                    while x == 0:
                        firstlist = seqlist
                        x = x + 1
        else:
            fdic_order = 0
                    
        # also the one above should be given a big boost if ALL found together 
          
        # could make another metric for if they are not next to each other but still close 
        
        return(searchsentence, words, fullcount_order, combocount_order, fullidf_order, fdic_order)
    
    except:
        return("")



## Search examples
search('Full-genome phylogenetic analysis')[1]
search('phylogenetic analysis of full genome')[1]
search('Full-genome phylogenetic analysis')
search('genome')
search('Full-genome')
search('Fullgenome')
search('covid')
search('PCR')
search('pathogens')
search('GISAID')

# 0 return will give back the search term, the rest will give back metrics (see above)
# Save metrics to dataframe for use in ranking and machine learning 
result1 = search('Full-genome phylogenetic analysis')
result2 = search('phylogenetic analysis of full genome')
result3 = search('Virus genome analysis')
result4 = search('genome')
result5 = search('phylogenetic')
result6 = search('evolutionary relationship')
result7 = search('pathogens')
result8 = search('GISAID')
df_search = pd.DataFrame([result1, result2, result3, result4, result5, result6, result7, result8])
# TODO check column names with respect to output of search function
df_search.columns = ['search term', 'actual_words_searched','num_occur','percentage_of_terms','td-idf','word_order']
df_search

df_search.to_excel("Data/output/search_results.xlsx")

# Look to see if the top documents seem to make sense
alldocslist[1]

## Double check: search in original df
df[df['Raw_Text'].str.contains('genome')] 
df[df['Raw_Text'].str.contains('phylogenetic')] 
df[df['Raw_Text'].str.contains('Covid')]  # only 9 papers


# =============================================================================
# PART V: Rank and return (rule based)
# =============================================================================
# Create a simple (non-machine learning) rank and return function

def rank(term):
    results = search(term)
    # print(results)
    # get metrics 
    num_score = results[2]
    per_score = results[3]
    tfscore = results[4]
    order_score = results[5]
    
    final_candidates = []

    # rule1: if high word order score & 100% percentage terms then put at top position
    try:
        first_candidates = []

        for candidates in order_score:
            if candidates[1] > 1:
                first_candidates.append(candidates[0])

        second_candidates = []

        for match_candidates in per_score:
            if match_candidates[1] == 1:
                second_candidates.append(match_candidates[0])
            if match_candidates[1] == 1 and match_candidates[0] in first_candidates:
                final_candidates.append(match_candidates[0])

    # rule2: next add other word order score which are greater than 1 
        t3_order = first_candidates[0:3]
        for each in t3_order:
            if each not in final_candidates:
                final_candidates.insert(len(final_candidates),each)

    # rule3: next add top td-idf results
        final_candidates.insert(len(final_candidates),tfscore[0][0])
        final_candidates.insert(len(final_candidates),tfscore[1][0])

    # rule4: next add other high percentage score 
        t3_per = second_candidates[0:3]
        for each in t3_per:
            if each not in final_candidates:
                final_candidates.insert(len(final_candidates),each)

    #rule5: next add any other top results for metrics
        othertops = [num_score[0][0],per_score[0][0],tfscore[0][0],order_score[0][0]]
        for top in othertops:
            if top not in final_candidates:
                final_candidates.insert(len(final_candidates),top)
                
    # unless single term searched, in which case just return 
    except:
        othertops = [num_score[0][0],num_score[1][0],num_score[2][0],per_score[0][0],tfscore[0][0]]
        for top in othertops:
            if top not in final_candidates:
                final_candidates.insert(len(final_candidates),top)

    print('\nRanked papers (document number):', final_candidates,)

    for index, results in enumerate(final_candidates):
        if index < 5:

            print('\n\nRESULT {}:'. format(index + 1), df.title[results])

            search_results = search_sentence(results, term)
            print('\nSentences:', search_results)

            print('\nPaper ID:', df.paper_id[results], '(Document no: {})'. format(results))
            print('\nAuthors:', df.authors[results], '\n')
            # print('Affiliations', df.affiliations[results])
            print(df.abstract[results]) # Abstract

    # return(final_candidates, results)

# Find sentence of search word(s)
def search_sentence(doc_number, search_term):
    sentence_index = []
    search_list = search_term.split()
    for sentence in df.Sentences[doc_number]:
        for search_word in search_list:
            if search_word.lower() in sentence.lower():
                sentence_index.append(sentence) # df.Sentences[doc_number].index(sentence)
    return sentence_index


# Check
search_term = 'Full-genome phylogenetic'
search_term = 'Sustainable risk reduction strategies'
results = search(search_term)
print(results)
doc_number = 1
search_sentence(doc_number, search_term)

search_term = 'genome'
# search_term = 'Full-genome'
results = search(search_term)
print(results)
doc_number = 1
search_sentence(doc_number, search_term)


## Examples
# Search return(searchsentence,words,fullcount_order,combocount_order,fullidf_order,fdic_order)
search('Full-genome phylogenetic analysis')
search('Full-genome phylogenetic')
search('genome')
search('covid')
search('PCR')
search('pathogens')
search('GISAID')

# Rank
rank('Full-genome phylogenetic analysis')
rank('Full-genome phylogenetic')
rank('phylogenetic analysis of full genome')
rank('Full-genome phylogenetic analysis')
rank('genome')
rank('covid')
rank('PCR')
rank('pathogens')
rank('GISAID')


# =============================================================================
# CASE 0: Sustainable risk reduction strategies
# =============================================================================
search_case_7 = search('Sustainable risk reduction strategies') 
search('Sustainable risk reduction strategies')
rank('Sustainable risk reduction strategies')

