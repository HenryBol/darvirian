# Hackaton for Peace, Justice and Security
# June 14th - 16th, 2019
# Challenge: National Rapporteur
# Team: NRteam1

# Note: all code and data is stored on Github

# Goal: Search tool for National Rapporteur

# Contents:
# PART I: Preparing the documents/webpages: text cleaning and creating 'alldoclist'(all 75 PDFs text content) and 'plot_data' (3D-list with all PDFs with tokenized words)
# PART II: CREATING THE INVERSE-INDEX: creates 'worddic' (dict with all words: vectorized and in which document(s) it occurs on which position(s) and tfidf)
# PART III: The Search Engine: function 'search'
# PART IV: Rank and return (rules based): function 'rank' based on 5 rules and providing summaries
# CASE I: NR: get all information on a certain topic; e.g. prostitution
# CASE II: Student: get all information about victims/ perpetrators
# CASE III: Media: has NR a point of view taking-in a passport of a delinquent of sexual violence against children so that he/ she is not able to travel.
# PART V: Rank and return (machine learning) - Work in Progress

# Next Actions:
# Discuss with National Rapporteur team (Robbert) to hand-over all data and code
# Input data row 66 and 67 have a NaN; temporarily changed to '' by check on NaN
# Check stemming
# one file is *.doc: to be included
# Part V Rank and return (machine learning): not ready yet: error message IndexError: list index out of range

# Notes:
# The website contains 190 PDFs; the number of provided PDFs is 75 only -> all 190 are scraped and used to have the full data set
# Also all webpages ('summaries') are used in the search presentation
# E.g. on the website a PDF is called 'Nationaal Rapporteur Mensenhandel en Seksueel Geweld tegen Kinderen (2016). Prostitutie en mensenhandel'
# in the provided PDFs this PDF is 'Prostitutie en mensenhandel.pdf' (no 43)

# Nice to have:
# Publications in English (e.g. 11 and 12)
# Check 38 and 39 are the same docs (title is a bit different only) - check for more

# Code is based on: https://www.kaggle.com/amitkumarjaiswal/nlp-search-engine


# =============================================================================
# PART I: Preparing the documents/webpages
# =============================================================================
## Importing the libraries
import pandas as pd
import numpy as np 
import string
#import random
import re

#import nltk
#from nltk.corpus import brown
#from nltk.corpus import reuters

from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
#from nltk.tokenize import RegexpTokenizer

from nltk.corpus import stopwords

#from nltk.stem.porter import PorterStemmer
#from nltk.stem import SnowballStemmer


# =============================================================================
# PART Ia: Preprocessing
# =============================================================================
## Read file with all PDF content (as processed by PyPDF2)
#df = pd.read_excel(r'output\pdf_content.xlsx') # file with all the 75 PDFs as provided (Slack)
#len(df) # 75 Publications

#df = pd.read_excel(r'output\pdfminer190_content.xlsx') # file with all 190 PDFs as scraped from the website
#df = pd.read_excel(r'output\pdfminer190_content_per_page.xlsx') # file with all 190 PDFs as scaped from the website
#df = pd.read_csv(r'output\pdfminer190_content_per_page.csv') # file with all 190 PDFs as scaped from the website

#df = pd.read_pickle('output/pdfminer190_content_per_page.pkl')
#df = pd.read_pickle('output/pdfminer190_content_per_page-2.pkl')
#df = pd.read_pickle('output/pdfminer190_content_per_page-166.pkl')
df = pd.read_pickle('output/pdfminer190_content_per_page_190619.pkl')
len(df) # 190 Publications


# Check NaNs
df.isnull().values.any()
df.isna().any() # Raw_Text
NaN_list_rows = df.isnull().sum(axis=1).sort_values(ascending = False)
df = df.replace(np.nan, '', regex=True)

# Clean text; check add characters e.g. '-' add or not (also used as hyphen)
# df.Raw_Text = [re.sub(r"[^a-zA-Z ]", "", file) for file in df] 
# df.Raw_Text = [re.sub(r"[^a-zA-Z0-9 -]", "", file) for file in df.Raw_Text] 
# HB 290320 the above lines gave an error; I changed dit to:
df.Raw_Text = [re.sub(r'[^a-zA-Z ]', '', str(x)) for x in df.Raw_Text]


# Check: search in original df (PyPDF2 bug)
#df[df['Raw_Text'].str.contains('slachtoer')] 
#df[df['Raw_Text'].str.contains('-')] 


## Extra pre-processing with PyPDF2 exctract
# 'slachtoffer' is mis-extracted as 'slachtoer' by PyPDF2
#df['Raw_Text'] = df['Raw_Text'].str.replace('slachtoer', 'slachtoffer')


# =============================================================================
# PART Ib: Preprocessing from example code
# =============================================================================
# Remove punctuation from all DOCs and create alldocslist
#exclude = set(string.punctuation)
alldocslist = []
alldocslist = df.Raw_Text
alldocslist = alldocslist.tolist()
alldocslist_cleanlist = alldocslist.copy()

# Joining pages of one doc
# HB 290320 the code below turns words into seperate characters, therefore, I disabled this code part
# pagess = alldocslist.copy()
# #a = [' '.join(pages) for pages in pagess]
# a = [' '.join(pages) for pages in pagess if pages is not None]
# alldocslist = a.copy()

df['Sentences'] = None

for i in range(len(alldocslist)):
    sent_text = sent_tokenize(alldocslist[i])
    df.Sentences[i] = sent_text

# Remove punctuation
alldocslist = ["".join( j for j in i if j not in string.punctuation) for i in alldocslist]

#exclude = set(string.punctuation)
##alldocslist = []
#i = 0
#for i in range(len(df)):
#    text = df.Raw_Text[i]
#    text = ''.join(ch for ch in text if ch not in exclude)
#    alldocslist.append(text)
#print(alldocslist[1])

# Tokenize words in all DOCS 
plot_data = [[]] * len(alldocslist)
index = 0
for doc in alldocslist:
    text = doc
    tokentext = word_tokenize(text)
    plot_data[index].append(tokentext)
    index +=1
print(plot_data[0][1])

print(plot_data[1][2])

##### check Reuters example for structure #####
#len(reuters.fileids())
#reuters.raw(fileids=['test/14826'])[0:201]
## remove punctuation from all DOCs 
#exclude = set(string.punctuation)
#alldocslist2 = []
#for index, i in  enumerate(reuters.fileids()):
#    text = reuters.raw(fileids=[i])
#    text = ''.join(ch for ch in text if ch not in exclude)
#    alldocslist2.append(text)
#print(alldocslist2[1])
##tokenize words in all DOCS 
#plot_data2 = [[]] * len(alldocslist2)
#for doc in alldocslist2:
#    text = doc
#    tokentext = word_tokenize(text)
#    plot_data2[index].append(tokentext)
#print(plot_data2[0][1])
##### conclusion: same structure #####

# Make all words lower case for all docs 
for x in range(len(alldocslist)):
    lowers = [word.lower() for word in plot_data[0][x]]
    plot_data[0][x] = lowers
plot_data[0][1][0:10]

# Remove stop words from all docs 
stop_words = set(stopwords.words('dutch'))
#stopwords = stopwords.union(set(['een','van'])) # add extra stopwords
for x in range(len(alldocslist)):
    filtered_sentence = [w for w in plot_data[0][x] if not w in stop_words]
    plot_data[0][x] = filtered_sentence
plot_data[0][1][0:10]


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
# PART II: CREATING THE INVERSE-INDEX
# =============================================================================
# Create inverse index which gives document number for each document and where word appears
# First we need to create a list of all words 
l = plot_data[0]
flatten = [item for sublist in l for item in sublist]
words = flatten
wordsunique = set(words)
wordsunique = list(wordsunique)
len(wordsunique)

# Create functions for TD-IDF / BM25
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

# Create dictonary of words
# THIS ONE-TIME INDEXING IS THE MOST PROCESSOR-INTENSIVE STEP AND WILL TAKE TIME TO RUN (BUT ONLY NEEDS TO BE RUN ONCE)
plottest = plot_data[0][0:1000]

worddic = {}
for doc in plottest:
    for word in wordsunique:
        if word in doc:
            word = str(word)
            index = plottest.index(doc)
            positions = list(np.where(np.array(plottest[index]) == word)[0])
            idfs = tfidf(word,doc,plottest)
            try:
                worddic[word].append([index,positions,idfs])
            except:
                worddic[word] = []
                worddic[word].append([index,positions,idfs])
# The index creates a dictionary with each word as a KEY and a list of doc indexes, word positions, and td-idf score as VALUES
worddic['diefstal']
worddic['kind']
worddic['jurisprudentie']
worddic['mishandeling']
worddic['prostitutie']
worddic['aruba']

## Pickle (save) the dictonary to avoid re-calculating
#np.save('output/worddic_1000_210619-189.npy', worddic, allow_pickle=True)
np.save('output/worddic_a-z_270619-189.npy', worddic, allow_pickle=True)

# or via pickle:
import pickle
f = open("output/worddic_a-z_270619-189.pkl","wb")
pickle.dump(worddic,f)
f.close()

## Load pickle file worddic
worddic = np.load('output/worddic_1000_210619-189.npy') # this gives an ndarray object 
# or via pickle:
pickle_in = open("output/worddic_1000_210619-189.pkl","rb")
worddic = pickle.load(pickle_in) 

worddic.write_to_excel('output/worddic_a-z_270619-189.xlsx') # not possible to write dict to excel
worddic.write_to_csv('output/worddic_a-z_270619-189.csv') # not possible to write dict to csv


## Split dictionary into keys and values 
keys = worddic.keys() 
values = worddic.values() 
items = worddic.items()
worddic_list = list(worddic.items())

help(items)

# printing keys and values seperately 
print("keys : ", str(keys)) 
print("values : ", str(values)) 

# check first and last keys
worddic_list[0]
worddic_list[-1]

# Extract from dictionary (probably not the most intelligent way to code - but it works)
# https://stackoverflow.com/questions/4194365/python-how-to-get-a-subset-of-dict
def get_range(dictionary, begin, end):
  return {k: v for k, v in dictionary.items() if begin <= k <= end}

worddic_09 = get_range(worddic, '0', 'a')
worddic_a = get_range(worddic, 'a', 'b')
worddic_b = get_range(worddic, 'b', 'c')
worddic_c = get_range(worddic, 'c', 'd')
worddic_d = get_range(worddic, 'd', 'e')
worddic_e = get_range(worddic, 'e', 'f')
worddic_f = get_range(worddic, 'f', 'g')
worddic_g = get_range(worddic, 'g', 'h')
worddic_h = get_range(worddic, 'h', 'i')
worddic_i = get_range(worddic, 'i', 'j')
worddic_j = get_range(worddic, 'j', 'k')
worddic_k = get_range(worddic, 'k', 'l')
worddic_l = get_range(worddic, 'l', 'm')
worddic_m = get_range(worddic, 'm', 'n')
worddic_n = get_range(worddic, 'n', 'o')
worddic_o = get_range(worddic, 'o', 'p')
worddic_p = get_range(worddic, 'p', 'q')
worddic_q = get_range(worddic, 'q', 'r')
worddic_r = get_range(worddic, 'r', 's')
worddic_s = get_range(worddic, 's', 't')
worddic_t = get_range(worddic, 't', 'u')
worddic_u = get_range(worddic, 'u', 'v')
worddic_v = get_range(worddic, 'v', 'w')
worddic_w = get_range(worddic, 'w', 'x')
worddic_x = get_range(worddic, 'x', 'y')
worddic_y = get_range(worddic, 'y', 'z')

# special case z
def get_range2(dictionary, begin, end):
  return {k: v for k, v in dictionary.items() if begin <= k < end}
worddic_z = get_range2(worddic, 'z', 'zűhlke')

# all others - NOTE the caharcter 'λλ^' is not the last oen - so should be improved
def get_range3(dictionary, begin, end):
  return {k: v for k, v in dictionary.items() if begin < k <= end}
worddic_other = get_range3(worddic, 'zűhlke', 'λλ^')

# Check on values - however in order as worddic as dict
#list(worddic.keys())[0]
#list(worddic.keys())[-1]
#list(worddic_09.keys())[0]
#list(worddic_09.keys())[-1]
#list(worddic_a.keys())[0]
#list(worddic_a.keys())[-1]
#help(worddic)

# Save dictionaries to file
np.save('output/worddic/worddic_09', worddic_09, allow_pickle=True)
np.save('output/worddic/worddic_a', worddic_a, allow_pickle=True)
np.save('output/worddic/worddic_b', worddic_b, allow_pickle=True)
np.save('output/worddic/worddic_c', worddic_c, allow_pickle=True)
np.save('output/worddic/worddic_d', worddic_d, allow_pickle=True)
np.save('output/worddic/worddic_e', worddic_e, allow_pickle=True)
np.save('output/worddic/worddic_f', worddic_f, allow_pickle=True)
np.save('output/worddic/worddic_g', worddic_g, allow_pickle=True)
np.save('output/worddic/worddic_h', worddic_h, allow_pickle=True)
np.save('output/worddic/worddic_i', worddic_i, allow_pickle=True)
np.save('output/worddic/worddic_j', worddic_j, allow_pickle=True)
np.save('output/worddic/worddic_k', worddic_k, allow_pickle=True)
np.save('output/worddic/worddic_l', worddic_l, allow_pickle=True)
np.save('output/worddic/worddic_m', worddic_m, allow_pickle=True)
np.save('output/worddic/worddic_n', worddic_n, allow_pickle=True)
np.save('output/worddic/worddic_o', worddic_o, allow_pickle=True)
np.save('output/worddic/worddic_p', worddic_p, allow_pickle=True)
np.save('output/worddic/worddic_q', worddic_q, allow_pickle=True)
np.save('output/worddic/worddic_r', worddic_r, allow_pickle=True)
np.save('output/worddic/worddic_s', worddic_s, allow_pickle=True)
np.save('output/worddic/worddic_t', worddic_t, allow_pickle=True)
np.save('output/worddic/worddic_u', worddic_u, allow_pickle=True)
np.save('output/worddic/worddic_v', worddic_v, allow_pickle=True)
np.save('output/worddic/worddic_w', worddic_w, allow_pickle=True)
np.save('output/worddic/worddic_x', worddic_x, allow_pickle=True)
np.save('output/worddic/worddic_y', worddic_y, allow_pickle=True)
np.save('output/worddic/worddic_z', worddic_z, allow_pickle=True)
np.save('output/worddic/worddic_other', worddic_other, allow_pickle=True)


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


# Save dictionary to file - not working
# https://pythonspot.com/save-a-dictionary-to-a-file/
#import csv
#w = csv.writer(open("output/worddic_1000_250619-189.csv", "w"))
#for key, val in worddic2.items():
#    w.writerow([key, val])

# Convert to dataframe - not working
# df_worddic = pd.DataFrame.from_dict(worddic) - error message
# https://stackoverflow.com/questions/20763012/creating-a-pandas-dataframe-from-a-numpy-array-how-do-i-specify-the-index-colum
#df_worddic = pd.DataFrame(data = worddic[1:,1:],    # values
#                          index = worddic[1:,0],    # 1st column as index
#                          columns = worddic[0,1:])  


# =============================================================================
# PART III: The Search Engine
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
search('kinderen diefstal kenmerken Oost-Europa')[1]
search('kinderen diefstal kenmerken Oost-Europa')
search('prostitutie')[1]

# 0 return will give back the search term, the rest will give back metrics (see above)
# Save metrics to dataframe for use in ranking and machine learning 
result1 = search('kinderen diefstal kenmerken Oost-Europa')
result2 = search('kinderen')
result3 = search('diefstal')
result4 = search('kenmerken')
result5 = search('Oost-Europa') # hyphen is not included in text
result6 = search('uitbuiting')
result7 = search('ouders')
result8 = search('prostitutie')
df_search = pd.DataFrame([result1,result2,result3,result4,result5,result6,result7,result8])
df_search.columns = ['search term', 'actual_words_searched','num_occur','percentage_of_terms','td-idf','word_order']
df_search

df_search.to_excel("output/search_results.xlsx")

# Look to see if the top documents seem to make sense
alldocslist[1]

## Double check: search in original df
df[df['Raw_Text'].str.contains('diefstal')] 
# ok (file 37 and 38 are the same)

df.columns

# =============================================================================
# PART IV: Rank and return (rules based)
# =============================================================================
# Create a simple (non-machine learning) rank and return function

def rank(term):
    results = search(term)
    print(results)
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
                
    for index, results in enumerate(final_candidates):
        if index < 5:
            print('\nfinal candidates:', final_candidates)
#            print("RESULT", index + 1, ":", alldocslist[results][0:100],"...")
            print("RESULT", index + 1, ":", df.Title[results])

            search_results = search_sentence(results, term)
            print("Sentences:", search_results)
            print(df.URL[results]) # link to PDF document in directory

            page = df.webpage[results] # get the publication webpage ('summary article') of where the PDF is located as a download
            print('Samenvatting:', df_webpages.type[page], df_webpages.date[page], df_webpages.header[page], df_webpages.intro[page], df_webpages.url[page])

df_pdfs = pd.read_excel(r'output\all_PDFs.xlsx') # file with all 190 PDFs as scraped from the website
# Add mapping column in df to df_pdfs
for i in range(len(df_pdfs)):
    string = df_pdfs.urls[i].rpartition('/')[2] # title of PDF only
    df_pdfs.loc[i,'url_title'] = string
    pointer = df.loc[df['PDF_name'] == string].index
    df.loc[pointer, 'Indexes_to_df_pdfs'] = i

# Include all data from df_pdfs in df
for i in range(len(df)):
    j = df.loc[i, 'Indexes_to_df_pdfs']
    df.loc[i, 'Title'] = df_pdfs.titles[j]
    df.loc[i, 'URL'] = df_pdfs.urls[j]
    df.loc[i, 'webpage'] = df_pdfs.webpage[j]

df_webpages = pd.read_excel(r'output\website_content.xlsx') # file with all 139 webpages as scraped from the website
## Include all data from df_webpages in df
#i = 0
#for i in range(len(df)):
#    j = df.loc[i, 'webpage'].astype(int)
#    list_webpage = df_webpages.iloc[j]
##    df.iloc[i, 'Webpage'] = list_webpage
#
#df.at[i, 'Webpage'] = list_webpage.astype(object)
#df.columns
#df = df[df.columns[:-1]]
#type(list_webpage)
#print(list_webpage)

df.columns
df_webpages.columns
df['Meta_Data'][0]

def search_sentence(doc_number, search_term):
    sentence_index = []
    for sentence in df.Sentences[doc_number]:
        if search_term.lower() in sentence.lower():
            sentence_index.append(sentence) # df.Sentences[doc_number].index(sentence)
    return sentence_index

### 
# check on function seach_sentence
###
# see https://stackoverflow.com/questions/40088559/python-check-if-two-words-are-in-a-string
    
#list_check = search_sentence(95, 'trauma')
#list_check = search_sentence(35, 'kinderen')
#list_check = search_sentence(99, 'Robert M.')
list_check = search_sentence(14, 'kinderen diefstal')
search_term = 'herdenkingen verleden heden'
results = search(search_term)
print(results)
doc_with_sentences = df.Sentences[26]
doc_number = 26
#search_term = 'heden' 
search_zinnen = search_sentence(26, search_term)

#sentence_index = []
#for sentence in df.Sentences[doc_number]:
#    print(sentence)
#    if search_term.lower() in sentence.lower():
#        sentence_index.append(sentence) # df.Sentences[doc_number].index(sentence)


## Example of output 
# return(searchsentence,words,fullcount_order,combocount_order,fullidf_order,fdic_order)
search('kinderen diefstal kenmerken Oost-Europa')
search('kinderen')
search('Oost-Europa')
search('Robert M.')
search('Vera')
search('NaN')
search('kinderen diefstal')

rank('kinderen diefstal kenmerken Oost-Europa')
rank('kinderen diefstal')
rank('misbruik diefstal')
rank('misbruik kinderen')
rank('geweld opsporing ')
rank('kinderen')
rank('kind')


rank('kinderen')
rank('aruba')
rank('trauma')
rank('Robert M.')
rank('Robert')
rank('Vera')
rank('OostEuropa')
rank('Robert misbruik')

rank('herdenkingen verleden heden')


# Example
search_term = 'Robert'
rank(search_term)
# final candidates (i.e. top5): [68, 161, 65, 11, 14]
# 68 highest ranking PDF: 
# RESULT 1 : First Report on Child Pornography
# Sentences: ['Also known as the “Robert M.” case, after the chief suspect.', '351 \n\nSee also 3.7.5 regarding the impact upon and support provided to parents of victims in the sexual abuse \ncases of ‘Benno L.’ and ‘Robert M.’ (the Amsterdam sexual abuse case).', '363 The extent of these \n\n356 \n\n357 \n\n358 \n\n359 \n360 \n361 \n362 \n363 \n\nRobert M. zeven weken in Pieter Baan Centrum [Robert M. to stay seven weeks in Pieter Baan Centrum], nu.nl, \n5 August 2011.', 'It is notable that little attention has been devoted to the child pornography aspect of the sexual abuse \ncommitted by Robert M.\nParliamentary Papers II 2005/06, 29 326, no.6.', 'As a result \nof the public reactions to major abuse cases, such as those surrounding the cases of ‘Robert M.’ and \n‘Benno L.’, paedophiles can end up being forced into even greater isolation95.', 'The work of the sex offences di-\nvision in Amsterdam included analysing the (child abuse) images found, in order to identify the victims, \nand for the purpose of the (detection and) prosecution of Robert M. and his partner.', 'There were sufficient indications present to suggest that Robert M. was maintaining a network with \nlike-minded individuals.', 'The analysis of the e-mail and chatroom traffic was used to \nreveal the individuals with whom Robert M. had contact, and what had been exchanged.', 'The National Crime \nSquad of the National Police Services Agency (KLPD) assembled a multi-disciplinary investigation \nteam in order to map out the (international) network of the suspected sex offender Robert M. The \ninvestigation team consisted of digital technology experts from the National Crime Squad, the \nSpecialist Criminal Investigations Applications Department (DSRT), the Amsterdam-Amstelland \npolice force, sex offence specialists from the Department of International Police Information, and \ndetectives from other KLPD services.', '180\n\nChild PornograPhy \n\nFirst report oF the Dutch NatioNal rapporteur\n\nThe Amsterdam sexual abuse case surrounding Robert M. is a recent example of the above-\nmentioned problem.']
# Link to PDF https://www.nationaalrapporteur.nl/binaries/child-pornography-tcm64-426462_tcm23-34786.pdf
# Summary: webpage ('summary article') that contains download link to pdf 68
# Samenvatting: Rapport 12-10-2011 Eerste Rapportage Kinderpornografie  
# Kinderpornografie is seksueel geweld tegen kinderen en moet op geïntegreerde wijze worden aangepakt. Dat is de belangrijkste conclusie van de Nationaal Rapporteur naar aanleiding van haar onderzoek naar kinderpornografie.
# https://www.nationaalrapporteur.nl/Publicaties/Eersterapportagekinderpornografie/index.aspx


# =============================================================================
# CASE 0: NR: expert looking for info to verify assumption
# =============================================================================
search_Case_0 = search('kinderen diefstal kenmerken OostEuropa') # note hyphen is not included in text
rank('kinderen diefstal kenmerken OostEuropa')


# =============================================================================
# CASE I: NR: get all information on a certain topic; e.g. prostitution
# =============================================================================
search_Case_I = search('prostitutie')
rank('prostitutie')


# =============================================================================
# CASE II: Student: get all information about victims/ perpetrators
# =============================================================================
search_Case_II = search('slachtoffer dader') 
rank('slachtoffer dader')


# =============================================================================
# CASE III: Media: has NR a point of view taking-in a passport of a delinquent 
# of sexual violence against children so that he/ she is not able to travel.
# =============================================================================
search_Case_III = search('paspoort delinquent sexueel misbruik kinderen')
rank('paspoort delinquent sexueel misbruik kinderen')


# =============================================================================
# Save datafram to pickle
# =============================================================================
df.to_pickle('output/df210619.pkl')
df_webpages.to_pickle('output/df_webpages210619.pkl')


# =============================================================================
# PART V: Rank and return (machine learning) - not ready yet
# =============================================================================
# Create pseudo-truth set using first 5 words 
# Because I don't have a turth set I will generate a pseudo one by pulling terms from the documents - this is far from perfect
# as it may not approximate well peoples actual queries but it will serve well to build the ML architecture 
#df_truth = pd.DataFrame()
#
#for doc in plottest:
#    first_five = doc[0:5]
#    test_sentence = ' '.join(first_five)
#    result = search(test_sentence)
#    df_temp = pd.DataFrame([result])
#    df_truth= pd.concat([df_truth, df_temp])
#
#df_truth['truth'] = range(0,len(plottest))
#
## Create another psuedo-truth set using random 3 word sequence from docs
#df_truth1 = pd.DataFrame()
#seqlen = 3
#
#for doc in plottest:
#    try:
#        start = random.randint(0,(len(doc)-seqlen))
#        random_seq = doc[start:start+seqlen]
#        test_sentence = ' '.join(random_seq)
#    except:
#        test_sentence = doc[0]
#    result = search(test_sentence)
#    df_temp = pd.DataFrame([result])
#    df_truth1= pd.concat([df_truth1, df_temp])
#
#df_truth1['truth'] = range(0,len(plottest))
#
#
## Create another psuedo-truth set using different random 4 word sequence from docs
#df_truth2 = pd.DataFrame()
#seqlen = 4
#
#for doc in plottest:
#    try:
#        start = random.randint(0,(len(doc)-seqlen))
#        random_seq = doc[start:start+seqlen]
#        test_sentence = ' '.join(random_seq)
#    except:
#        test_sentence = doc[0]
#    result = search(test_sentence)
#    df_temp = pd.DataFrame([result])
#    df_truth2= pd.concat([df_truth2, df_temp])
#
#df_truth2['truth'] = range(0,len(plottest))
#
## Create another psuedo-truth set using different random 2 word sequence from docs
#df_truth3 = pd.DataFrame()
#seqlen = 2
#
#for doc in plottest:
#    try:
#        start = random.randint(0,(len(doc)-seqlen))
#        random_seq = doc[start:start+seqlen]
#        test_sentence = ' '.join(random_seq)
#    except:
#        test_sentence = doc[0]
#    result = search(test_sentence)
#    df_temp = pd.DataFrame([result])
#    df_truth3= pd.concat([df_truth3, df_temp])
#
#df_truth3['truth'] = range(0,len(plottest))
#
## Combine the truth sets and save to disk 
#truth_set = pd.concat([df_truth,df_truth1,df_truth2,df_truth3])
#truth_set.columns = ['search term', 'actual_words_searched','num_occur','percentage_of_terms','td-idf','word_order','truth']
#truth_set.to_csv("output/truth_set_final.csv")
#
#truth_set[0:10]
#
#truth_set
#test_set = truth_set[0:3]
#test_set
#
#
## convert to long format for ML 
## WARNING AGAIN THIS IS A SLOW PROCESS DUE TO RAM ILOC - COULD BE OPTIMISED FOR FASTER PERFORMANCE 
## BUG When min(maxnum, len(truth_set) <- is a int not a list because of very short variable length)
#
## row is row
## column is variable
## i is the result 
#
#final_set =  pd.DataFrame()
#test_set = truth_set[1:100]
#maxnum = 5
#
#for row in range(0,len(test_set.index)):
#    test_set = truth_set[1:100]
#    for col in range(2,6):
#        for i in range(0,min(maxnum,len(truth_set.iloc[row][col]))):
#            x = pd.DataFrame([truth_set.iloc[row][col][i]])
#            x['truth'] = truth_set.iloc[row]['truth']
#            x.columns = [(str(truth_set.columns[col]),"index",i),(str(truth_set.columns[col]),"score",i),'truth']
#            test_set = test_set.merge(x,on='truth')
#    final_set = pd.concat([final_set,test_set])
#        
#final_set.head()
#Out[29]:
#
#final_set.to_csv("ML_set_100.csv")
#
#final_set2 = final_set.drop(['actual_words_searched','num_occur','percentage_of_terms','search term','td-idf','word_order'], 1)
#final_set2.to_csv("ML_set_100_3.csv")
#final_set2.head()
#
#final_set3 = final_set2
#final_set3[0:10]



