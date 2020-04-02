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


# =============================================================================
# Import the libraries
# =============================================================================
## Importing the libraries
import pandas as pd
import numpy as np 
import string
import re
import pickle

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
df = pd.read_csv('Data/CSV/biorxiv_clean.csv')
df.info()
df.columns
## Copy for convenience the data to a 'Raw_Text' column
df['Raw_Text'] = df['text']
## << TEMP CORD-19 DATA


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
# TODO check adding figures (0-9) which increases the number of unique word significantly (and slow the tdidf process)
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
# PART III: CREATING THE INVERSE-INDEX
# =============================================================================
# Create inverse index which gives document number for each document and where word appears

## Check unique words
all_words = [item for sublist in plot_data for item in sublist]
wordsunique = set(all_words)
wordsunique = list(wordsunique)
len(wordsunique)


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
# plottest = plot_data

# Create dictionary with a list as values
from collections import defaultdict
worddic = defaultdict(list)

# Loop (for reference and to make the comprehension (see below) a bit more understandable)
# for doc in plottest:
#     for word in set(doc): # set provides unique words in doc 
#         word = str(word)
#         index = plottest.index(doc)
#         positions = list(np.where(np.array(plottest[index]) == word)[0])
#         idfs = tfidf(word,doc,plottest)
#         worddic[word].append([index,positions,idfs])

# Create the dictionary via comprehension to speed up processing
# import time
# start = time.time()
# # TODO speed up
# [worddic[word].append([plottest.index(doc), list(np.where(np.array(plottest[plottest.index(doc)]) == word)[0]), tfidf(word,doc,plottest), ]) for doc in plottest for word in set(doc)]
# end = time.time()
# print(end - start) # duration 3.36 hours for full biorxiv_clean


## Save pickle file
# f = open("Data/output/worddic_biorxiv_clean_200402.pkl","wb")
# pickle.dump(worddic,f)
# f.close()

## Load pickle file worddic
pickle_in = open("Data/output/worddic_biorxiv_clean_200402.pkl","rb")
worddic = pickle.load(pickle_in) 

# Check
worddic['covid']
plottest[882][38]
len(worddic.keys())


## Split dictionary into keys and values 
keys = worddic.keys() 
values = worddic.values() 
items = worddic.items()
worddic_list = list(worddic.items())

# printing keys and values seperately 
print("keys : ", str(keys)) 
print("values : ", str(values)) 

# check first and last keys
worddic_list[0]
worddic_list[-1]

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

            # TODO check fdic order, seems always to be 0
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
        
        return(searchsentence,words,fullcount_order,combocount_order,fullidf_order,fdic_order)
    
    except:
        return("")

## Search    
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

# TODO CORD keywords
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
rank('Sustainable risk reduction strategies')
