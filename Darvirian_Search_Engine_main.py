# Kaggle challenge Covid-19
# Team: Darvirian
# Date: April 7th, 2020

# Contents:
# PART I: Load the data
# PART II: The Search Engine: function 'search'
# PART III: Rank and return (rules based): function 'rank' based on 5 rules and providing summaries
# PART IV: examples
# CASE 0: Sustainable risk reduction strategies

# Inspiration: https://www.kaggle.com/amitkumarjaiswal/nlp-search-engine

# =============================================================================
# Import the libraries
# =============================================================================
## Importing the libraries
import pandas as pd
import pickle
from collections import Counter


# =============================================================================
# PART I: LOAD THE DATA
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


## Load pickle file sentences
pickle_in = open("Data/output/sentences_200407.pkl","rb")
sentences = pickle.load(pickle_in) 

## Load pickle file plot_data
pickle_in = open("Data/output/plot_data_200407.pkl","rb")
plot_data = pickle.load(pickle_in) 

## Load pickle file df_dfidf (too big too store)
# pickle_in = open("Data/output/df_tfidf_200407.pkl","rb")
# df_tfidf = pickle.load(pickle_in) 

## Load pickle file worddic
pickle_in = open("Data/output/worddic_all_200407.pkl","rb")
worddic = pickle.load(pickle_in) 

# ## Split dictionary into keys and values 
# keys = worddic.keys() 
# values = worddic.values() 
# items = worddic.items()
# worddic_list = list(worddic.items())


# =============================================================================
# PART II: The Search Engine
# =============================================================================
# Create word search which takes multiple words (sentence) and finds documents that contain these words along with metrics for ranking:

# Output: searchsentence, words, fullcount_order, combocount_order, fullidf_order, fdic_order    
# (1) searchsentence: original sentence to be searched
# (2) words: words of the search sentence that are found in the dictionary (worddic)
# (3) fullcount_order: number of occurences of search words 
# (4) combocount_order: percentage of search terms
# (5) fullidf_order: sum of TD-IDF scores for search words (in ascending order)
# (6)) fdic_order: exact match bonus (# ()) Word ordering score )

# >>> example on limited dataset (first three docs of biorxiv))
# search('Full-genome phylogenetic analysis')
# (1) ('full-genome phylogenetic analysis',  # searchsentence: original searh sentence
# (2) ['phylogenetic', 'analysis'], # words: two of the search words are in the dictionary worddic
# (3) [(1, 7), (0, 1)], # fullcount_order: the search words (as found in dict) occur in total 7 times in doc 1 and 1 time in doc 0
# (4) [(1, 1.0), (0, 0.5)], # combocount_order: max value is 1, in doc 1 all searchwords (as in dict) are present (1), in doc 0 only 1 of the 2 search words are present (0.5)
# (5) [(1, 0.0025220519886750533), (0, 0.0005167452472220973)], # fullidf_order: doc 1 has a total (sum) tf-idf of 0.0025220519886750533, doc 0 a total tf-idf of 0.0005167452472220973
# (6) [(1, 1)]) # fdic_order:  
# <<<

searchsentence = 'Full-genome phylogenetic analysis'

def search(searchsentence):
    # remove try statements and change to if-else for speeding up search process (also in ranking)
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
        # realwords = []            
        # for word in words:
        #     if word in list(worddic.keys()):
        #         realwords.append(word)  
        words = [word for word in words if word in worddic.keys()]
        numwords = len(words)
        
        # metrics fullcount_order and fullidf_order: sum of number of occurences of all words in each doc (fullcount_order) and sum of TF-IDF score (fullidf_order)
        for word in words:
            # print(word)
            for indpos in worddic[word]:
                # print(indpos)
                index = indpos[0]
                amount = len(indpos[1])
                idfscore = indpos[2]
                # check if the index is already in the dictionary: add values to the keys
                if index in enddic.keys(): 
                    enddic[index] += amount
                    idfdic[index] += idfscore
                # if not, just make a two new keys and store the values    
                else:
                    enddic[index] = amount
                    idfdic[index] = idfscore
        fullcount_order = sorted(enddic.items(), key=lambda x:x[1], reverse=True)
        fullidf_order = sorted(idfdic.items(), key=lambda x:x[1], reverse=True) 

        # metric combocount_order: percentage of search words (as in dict) that appear in each doc
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
    
        # metric closedic: if words appear in same order as in search
        if len(words) > 1:
            # x = [] # per word: document(s) in which a word appears
            # y = [] # documents with more than one search word
            # for record in [worddic[z] for z in words]:
            #     for index in record:
            #           x.append(index[0])
            # all document(s) in which a search word appears (index[0] gives the document number)
            x = [index[0] for record in [worddic[z] for z in words] for index in record]
            # for i in x:               
            #     if x.count(i) > 1: # count number of times a doc occurs in x
            #         y.append(i)
            # y = list(set(y))
            # all document(s) in which more than one search word appears; keep unique values and sort
            y = sorted(list(set([i for i in x if x.count(i) > 1])))

# def diff(first, second):
#     second = set(second)
#     return [item for item in first if item not in second]

# diff(x,x2)
# diff(x2,x)


            # dictionary of documents and all positions (for docs with more than one search word in it)
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

# x = [index[0] for record in [worddic[z] for z in words] for index in record]

# closedic2 = {key_2: worddic[key_2] for key_2 in y}

# closedic2 = { your_key: old_dict[your_key] for your_key in your_keys }

# dict(zip(keys, [orig[k] for k in keys]))
# closedic2 = dict(zip(y, [worddic[k] for k in y]))

# worddic[1][]

# dict(worddic)

# passed = { key:value for key, value in marks.items() if value > 50 }
# closedic2 = { key:value for key, value in dict(worddic).items() if value in y}

# worddic['covid']

# items = worddic.items()
# items.keys(0)

# fdic_order[0]

# closedic = {k, func(v) for k, v in worddic.items() if k in y}

            # metric: fdic number of times search words appear in a doc in descending order
            # TODO check
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


# =============================================================================
# PART III: Rank and return (rule based)
# =============================================================================
# Create a simple rule based rank and return function

def rank(term):
    results = search(term)
    # get metrics 
    num_score = results[2] # number of search words (as in dict) in each doc (in ascending order)
    per_score = results[3] # percentage of search words (as in dict) in each doc (in ascending order; range [1,0)
    tfscore = results[4] # sum of tfidf of search words in each doc (in ascending order)
    order_score = results[5] # fidc order
    
    final_candidates = []

    # rule1: doc with high fidc order_score (>1) & 100% percentage search words (as in dict) on no. 1 position
    try:
        first_candidates = []

        # first candidate(s) comes from fidc order_score (with value > 1)
        for candidates in order_score: 
            if candidates[1] > 1:
                first_candidates.append(candidates[0])

        second_candidates = []

        for match_candidates in per_score:
            # if all words are in a document: add to second_candidates
            if match_candidates[1] == 1: 
                second_candidates.append(match_candidates[0])
            # if all words are in a document and this doc is also in first candidates: add to final_candidates
            if match_candidates[1] == 1 and match_candidates[0] in first_candidates:
                final_candidates.append(match_candidates[0])

    # rule2: add max 4 other worda with order_score greater than 1 (if not yet in final_candiates)
        t3_order = first_candidates[0:3]
        for each in t3_order:
            if each not in final_candidates:
                final_candidates.insert(len(final_candidates),each)

    # rule3: add 2 top td-idf results to final_candidates
        final_candidates.insert(len(final_candidates), tfscore[0][0])
        final_candidates.insert(len(final_candidates), tfscore[1][0])

    # rule4: next add other high percentage score (if not yet in final_candiates)
        t3_per = second_candidates[0:3] # the first 4 high precentages scores (if equal to 100% of search words in doc)
        for each in t3_per:
            if each not in final_candidates:
                final_candidates.insert(len(final_candidates),each)

    # rule5: next add any other no. 1 result in num_score, per_score, tfscore and order_score (if not yet in final_candidates)
        othertops = [num_score[0][0], per_score[0][0], tfscore[0][0], order_score[0][0]]
        for top in othertops:
            if top not in final_candidates:
                final_candidates.insert(len(final_candidates),top)
                
    # in case of a single term searched (as in dict): just return the following 5 scores
    except:
        othertops = [num_score[0][0], num_score[1][0], num_score[2][0], per_score[0][0], tfscore[0][0]]
        for top in othertops:
            if top not in final_candidates:
                final_candidates.insert(len(final_candidates),top)

    # print final candiates
    print('\nFound search words:', results[1])
    print('Ranked papers (document numbers):', final_candidates)
    
    # print max top 5 results: sentences with search words, paper iD (and documet number), authors and abstract
    for index, results in enumerate(final_candidates):
        if index < 5:

            print('\n\nRESULT {}:'. format(index + 1), df.title[results])
            print('\nPaper ID:', df.paper_id[results], '(Document no: {})'. format(results))
            print('\nAuthors:', df.authors[results])
            print('\n')
            print(df.abstract[results])
            search_results = search_sentence(results, term)
            print('Sentences:\n')
            print(search_results)

    # return(final_candidates, results)

# Find sentence of search word(s)ß
def search_sentence(doc_number, search_term):
    sentence_index = []
    search_list = search_term.split()
    # for sentence in df.Sentences[doc_number]:
    for sentence in sentences[doc_number]:
        for search_word in search_list:
            if search_word.lower() in sentence.lower():
                sentence_index.append(sentence) # df.Sentences[doc_number].index(sentence)
    return sentence_index


# =============================================================================
# PART IV: xamples
# =============================================================================
# Search return(searchsentence,words,fullcount_order,combocount_order,fullidf_order,fdic_order)
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
search('evolutionary relationship pathogens')
search('fatality rate')
search('Sustainable risk reduction strategies')

# Rank
rank('Full-genome phylogenetic analysis')
rank('Full-genome phylogenetic')
rank('genome phylogenetic')
rank('phylogenetic analysis of full genome')
rank('Full-genome phylogenetic analysis')
rank('genome')
rank('covid')
rank('PCR')
rank('pathogens')
rank('GISAID')
rank('virus spreading evolving')
rank('in-house singleplex assay')
rank('FTDRP multiplex RT-PCR')
rank('FTDRP multiplex RT-PCR Full-genome phylogenetic analysis')
rank('farmer')
rank('nagoya protocol')


# =============================================================================
# CASE 0: Sustainable risk reduction strategies
# =============================================================================
search_case_7 = search('Sustainable risk reduction strategies') 
search('Sustainable risk reduction strategies')
rank('Sustainable risk reduction strategies')

# =============================================================================
# CASE 1: Real-time tracking of whole genomes
# =============================================================================
rank('Real-time tracking of whole genomes and a mechanism for coordinating the rapid dissemination of that information to inform the development of diagnostics and therapeutics and to track variations of the virus over time.')


