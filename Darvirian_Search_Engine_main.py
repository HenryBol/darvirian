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
from collections import Counter
import re
import pickle
import pandas as pd


# =============================================================================
# PART I: LOAD THE DATA
# =============================================================================
## Read docs from CORD-19
# import os
# os.chdir("../Data/CSV")
# df_biorxiv = pd.read_csv('Data/CSV/biorxiv_clean.csv')
# df_clean_comm_use = pd.read_csv('Data/CSV/clean_comm_use.csv')
# df_clean_noncomm_use = pd.read_csv('Data/CSV/clean_noncomm_use.csv')
# df_clean_pmc = pd.read_csv('Data/CSV/clean_pmc.csv')

# # Add all dataframes togethers
# df = df_biorxiv.append(df_clean_comm_use).reset_index(drop=True)
# df = df.append(df_clean_noncomm_use).reset_index(drop=True)
# df = df.append(df_clean_pmc).reset_index(drop=True)

# df.columns

# Load snall version of df with papers
df = pd.read_csv('Data/output/df.csv')

## Load pickle file sentences
pickle_in = open('Data/output/sentences_200410.pkl', 'rb')
sentences = pickle.load(pickle_in)

## Load pickle file plot_data
# pickle_in = open('Data/output/plot_data_200407.pkl', 'rb')
# plot_data = pickle.load(pickle_in)

## Load pickle file worddic
pickle_in = open('Data/output/worddic_all_200410.pkl', 'rb')
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
# (6)) fdic_order: exact match bonus: word ordering score

# >>> example on limited dataset (first three docs of biorxiv))
# search('Full-genome phylogenetic analysis')
# (1) ('full-genome phylogenetic analysis',  # searchsentence: original search sentence
# (2) ['phylogenetic', 'analysis'], # words: two of the search words are in the dictionary worddic
# (3) [(1, 7), (0, 1)], # fullcount_order: the search words (as found in dict) occur in total 7 times in doc 1 and 1 time in doc 0
# (4) [(1, 1.0), (0, 0.5)], # combocount_order: max value is 1, in doc 1 all searchwords (as in dict) are present (1), in doc 0 only 1 of the 2 search words are present (0.5)
# (5) [(1, 0.0025220519886750533), (0, 0.0005167452472220973)], # fullidf_order: doc 1 has a total (sum) tf-idf of 0.0025220519886750533, doc 0 a total tf-idf of 0.0005167452472220973
# (6) [(1, 1)]) # fdic_order: doc 1 has once two search words next to each other
# <<<


def search(searchsentence):
    # split sentence into individual words
    searchsentence = searchsentence.lower()
    # split sentence in words and keep characters as in worddic
    words = searchsentence.split(' ')
    words = [re.sub(r'[^a-zA-Z.]', '', str(w)) for w in words]

    # temp dictionaries
    enddic = {}
    idfdic = {}
    closedic = {}

    # remove words if not in worddic (keep only the words that are in the dictionary)
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
    fullcount_order = sorted(enddic.items(), key=lambda x: x[1], reverse=True)
    fullidf_order = sorted(idfdic.items(), key=lambda x: x[1], reverse=True)

    # metric combocount_order: percentage of search words (as in dict) that appear in each doc
    # TODO check when combocountorder > 1 (and is it a reason to give these docs more relevance)
    alloptions = {k: worddic.get(k) for k in words}
    comboindex = [item[0] for worddex in alloptions.values() for item in worddex]
    combocount = Counter(comboindex) # count the time of each index
    for key in combocount:
        combocount[key] = combocount[key] / numwords
    combocount_order = sorted(combocount.items(), key=lambda x: x[1], reverse=True)

    # metric closedic: if words appear in same order as in search
    if len(words) > 1:
        # list with docs with a search word        
        x = [index[0] for record in [worddic[z] for z in words] for index in record]
        # list with docs with more than one search word
        y = sorted(list(set([i for i in x if x.count(i) > 1])))

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
        # Index add to comprehension:
        # closedic2 = [record[1] for wordbig in [worddic[x] for x in words] for record in wordbig if record[0] in y]

        # metric: fdic number of times search words appear in a doc in descending order
        # TODO check
        x = 0
        fdic = {}
        for index in y: # list with docs with more than one search word
            csum = []            
            for seqlist in closedic[index]:
                while x > 0:
                    secondlist = seqlist # second word positions
                    x = 0
                    # first and second word next to each other (in same order)
                    sol = [1 for i in firstlist if i + 1 in secondlist]
                    csum.append(sol)
                    fsum = [item for sublist in csum for item in sublist] 
                    fsum = sum(fsum) 
                    fdic[index] = fsum
                    fdic_order = sorted(fdic.items(), key=lambda x: x[1], reverse=True)
                while x == 0:
                    firstlist = seqlist # first word positions 
                    x = x + 1 
    else:
        fdic_order = 0

    # TODO another metric for if they are not next to each other but still close

    return(searchsentence, words, fullcount_order, combocount_order, fullidf_order, fdic_order)


# =============================================================================
# PART III: Rank and return (rule based)
# =============================================================================
# Create a simple rule based rank and return function

def rank(term):

    # get results from search
    results = search(term)
    # get metrics
    # TODO ADDED LINE
    search_words = results[1] # search words found in dictionary
    num_search_words = len(results[1]) # number of search words found in dictionary
    num_score = results[2] # number of search words (as in dict) in each doc (in descending order)
    per_score = results[3] # percentage of search words (as in dict) in each doc (in descending order)
    tfscore = results[4] # sum of tfidf of search words in each doc (in ascending order)
    order_score = results[5] # fidc order

    # list of documents in order of relevance
    final_candidates = []

    # no search term(s) not found
    if num_search_words == 0:
        print('Search term(s) not found')

    # single term searched (as in dict): return the following 5 scores
    if num_search_words == 1:
        num_score_list = [l[0] for l in num_score] # document numbers
        num_score_list = num_score_list[:3] # take max 3 documents from num_score
        num_score_list.append(per_score[0][0]) # add the best percentage score
        num_score_list.append(tfscore[0][0]) # add the best tf score
        final_candidates = list(set(num_score_list)) # remove duplicate document numbers


    # more than one search word (and found in dictionary)
    if num_search_words > 1:

        # rule1: doc with high fidc order_score (>1) & 100% percentage search words (as in dict) on no. 1 position
        first_candidates = []

        # first candidate(s) comes from fidc order_score (with value > 1)
        for candidates in order_score:
            if candidates[1] >= 1:
                first_candidates.append(candidates[0])

        second_candidates = []

        for match_candidates in per_score:
            # if all words are in a document: add to second_candidates
            # TODO check why per_score sometimes > 1 (change to >=1 ?)
            if match_candidates[1] == 1:
                second_candidates.append(match_candidates[0])
        # first final candidates have the highest score of search words next to each other and all search words (as in dict) in document  
        for match_candidates in first_candidates:
            if match_candidates in second_candidates:
                final_candidates.append(match_candidates)

        # rule2: add max 4 other words with order_score greater than 1 (if not yet in final_candiates)
        t3_order = first_candidates[0:3]
        for each in t3_order:
            if each not in final_candidates:
                final_candidates.insert(len(final_candidates), each)

        # rule3: add 2 top td-idf results to final_candidates
        final_candidates.insert(len(final_candidates), tfscore[0][0])
        final_candidates.insert(len(final_candidates), tfscore[1][0])

        # rule4: next add four other high percentage score (if not yet in final_candiates)
        t3_per = second_candidates[0:3] # the first 4 high percentages scores (if equal to 100% of search words in doc)
        for each in t3_per:
            if each not in final_candidates:
                final_candidates.insert(len(final_candidates), each)

        # rule5: next add any other no. 1 result in num_score, per_score, tfscore and order_score (if not yet in final_candidates)
        othertops = [num_score[0][0], per_score[0][0], tfscore[0][0], order_score[0][0]]
        for top in othertops:
            if top not in final_candidates:
                final_candidates.insert(len(final_candidates), top)


    # print final candidates
    print('\nFound search words:', results[1])
    print('Ranked papers (document numbers):', final_candidates)

    # top results: sentences with search words, paper ID (and documet number), authors and abstract
    df_results = pd.DataFrame(columns=['Title', 'Paper_id', 'Document_no', 'Authors', 'Abstract', 'Sentences', 'Search_words'])
    for index, results in enumerate(final_candidates):
        # if index < 5:
        df_results.loc[index+1, 'Title'] = df.title[results]
        df_results.loc[index+1, 'Paper_id'] = df.paper_id[results]
        df_results.loc[index+1, 'Document_no'] = results
        df_results.loc[index+1, 'Authors'] = df.authors[results]
        df_results.loc[index+1, 'Abstract'] = df.abstract[results]
        search_results = search_sentence(results, term)
        df_results.loc[index+1, 'Sentences'] = search_results
        
        # # TODO ADDED LINES
        # Find search words per document 
        df_results.loc[index+1, 'Search_words'] = [word for word in search_words for sub_list in sentences[results] if word in sub_list]

    return final_candidates, df_results


# Find sentence of search word(s)
def search_sentence(doc_number, search_term):
    sentence_index = []
    search_list = search_term.split()
    for sentence in sentences[doc_number]:
        for search_word in search_list:
            if search_word.lower() in sentence.lower():
                sentence_index.append(sentence) # df.Sentences[doc_number].index(sentence)
    return sentence_index


## print final candidates
# print('\nFound search words:', results[1])
# print('Ranked papers (document numbers):', final_candidates)

# df_results.loc[index, 'Title'] = df.title[results]
# print('\n\nRESULT {}:'. format(index + 1), df.title[results])
# df_results.loc[index, 'Paper_id'] = df.paper_id[results]
# df_results.loc[index, 'Document_no'] = results
# print('\nPaper ID:', df.paper_id[results], '(Document no: {})'. format(results))
# df_results.loc[index, 'Authors'] = df.authors[results]
# print('\nAuthors:', df.authors[results])
# print('\n')
# df_results.loc[index, 'Abstract'] = df.abstract[results]
# print(df.abstract[results])
# search_results = search_sentence(results, term)
# df_results.loc[index, 'Sentences'] = search_results
# print('Sentences:\n')
# print(search_results)


# =============================================================================
# PART IV: Examples
# =============================================================================
# Search return(searchsentence,words,fullcount_order,combocount_order,fullidf_order,fdic_order)
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


# Rank (disable return)
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


# tests with df_biorxiv only
term = 'Full-genome phylogenetic analysis'
rank('Full-genome phylogenetic analysis')
search('farmer')
search('Manhattan')
search('duties')
# df.text[36]

worddic['PCR']
worddic['covid']

searchsentence = 'Full-genome phylogenetic'
searchsentence = 'Full-genome phylogenetic analysis'
searchsentence = 'duties farmer'


# =============================================================================
# CASE 0: Sustainable risk reduction strategies
# =============================================================================
search_case_7 = search('Sustainable risk reduction strategies')
search('Sustainable risk reduction strategies')
rank('Sustainable risk reduction strategies')

papers, rank_result = rank('Sustainable risk reduction strategies')
rank_result.to_csv('Data/output/rank_result_0200410.csv')

Counter(rank_result.Search_words.iloc[0])


# =============================================================================
# CASE 1: Real-time tracking of whole genomes
# =============================================================================
rank('Real-time tracking of whole genomes and a mechanism for coordinating the rapid dissemination of that information to inform the development of diagnostics and therapeutics and to track variations of the virus over time.')

final_candidates, rank_result = rank('Real-time tracking of whole genomes and a mechanism for coordinating the rapid dissemination of that information to inform the development of diagnostics and therapeutics and to track variations of the virus over time.')

rank_result.to_csv('Data/output/rank_result_0200410.csv')

papers, rank_result = rank('Real-time tracking of whole genomes and a mechanism for coordinating the rapid dissemination of that information to inform the development of diagnostics and therapeutics and to track variations of the virus over time.')
rank_result.to_csv('Data/output/rank_result_0200410-2.csv')
    
    