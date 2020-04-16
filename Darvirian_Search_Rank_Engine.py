# Kaggle challenge Covid-19
# Team: Darvirian
# Developer: Henry Bol

# Contents:
# PART I: Load the data
# PART II: The Search Engine: function 'search'
# PART III: Rank and return (rules based): function 'rank' based on 5 rules and providing summaries
# PART IV: Function search sentence
# PART V: Function print result (ranked papers)
# PART VI: Examples 
# CASE 0: Sustainable risk reduction strategies

# Credits:
# Inspiration: https://www.kaggle.com/amitkumarjaiswal/nlp-search-engine

# =============================================================================
# Import the libraries
# =============================================================================
from collections import Counter
import re
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from wordcloud import WordCloud
# from PIL import Image


# =============================================================================
# PART I: Load the data
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

# Load small version of df with papers
pickle_in = open('Data/output/df.pkl', 'rb')
df = pickle.load(pickle_in)

# Load pickle file sentences
pickle_in = open('Data/output/sentences_200415.pkl', 'rb')
sentences = pickle.load(pickle_in)

# Load pickle file plot_data
# pickle_in = open('Data/output/plot_data_200407.pkl', 'rb')
# plot_data = pickle.load(pickle_in)

# Load pickle file worddic (word version)
# pickle_in = open('Data/output/worddic_all_200410.pkl', 'rb')
# worddic = pickle.load(pickle_in)

# Load pickle file worddic (numeric version)
# all words besides single characters:
# pickle_in = open('Data/output/worddic_all_200415_num.pkl', 'rb') 
# worddic = pickle.load(pickle_in)
# all words besides single characters and words that occur only once in all docs
pickle_in = open('Data/output/worddic_all_200415_num-2.pkl', 'rb')
worddic = pickle.load(pickle_in)

# Load pickle file word2idx
pickle_in = open('Data/output/word2idx_200415-2.pkl', 'rb')
word2idx = pickle.load(pickle_in)

# Load pickle file idx2word
pickle_in = open('Data/output/idx2word_200415-2.pkl', 'rb')
idx2word = pickle.load(pickle_in)

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
    words = [re.sub(r'[^a-zA-Z]', ' ', str(w)) for w in words]
    
    # remove words if not in worddic (keep only the words that are in the dictionary)
    words = [word for word in words if word in word2idx.keys()]
    numwords = len(words)
    
    # word2idx all words
    words = [word2idx[word] for word in words]

    # temp dictionaries
    enddic = {}
    idfdic = {}
    closedic = {}


    ## metrics fullcount_order and fullidf_order: sum of number of occurences of all words in each doc (fullcount_order) and sum of TF-IDF score (fullidf_order)
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


    ## metric combocount_order: percentage of search words (as in dict) that appear in each doc
    # TODO check when combocountorder > 1 (and is it a reason to give these docs more relevance)
    alloptions = {k: worddic.get(k) for k in words}
    comboindex = [item[0] for worddex in alloptions.values() for item in worddex]
    combocount = Counter(comboindex) # count the time of each index
    for key in combocount:
        combocount[key] = combocount[key] / numwords
    combocount_order = sorted(combocount.items(), key=lambda x: x[1], reverse=True)


    ## metric closedic: if words appear in same order as in search
    fdic_order = 0 # initialization in case of a single search word
    if len(words) > 1:
        # list with docs with a search word        
        x = [index[0] for record in [worddic[z] for z in words] for index in record]
        # list with docs with more than one search word
        # y = sorted(list(set([i for i in x if x.count(i) > 1])))
        counts = np.bincount(x)
        y = list(np.where([counts>1])[1])

        # dictionary of documents and all positions (for docs with more than one search word in it)
        # TODO speed up
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
        # TODO check comprehension:
        # closedic2 = {}
        # closedic2 = [record[1] for wordbig in [worddic[x] for x in words] for record in wordbig if record[0] in y]
        # TODO check defaultdict
        # closedic2 = defaultdict(list)
        # [closedic2[index].append(record[1]) for wordbig in [worddic[x] for x in words] for record in wordbig if record[0] in y]
        

        ## metric: fdic number of times search words appear in a doc in descending order
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

        ## TODO add metric search words in abstract
        ## TODO another metric for if they are not next to each other but still close
    
    
    ## idx2word all words
    words = [idx2word[word] for word in words]

    return(searchsentence, words, fullcount_order, combocount_order, fullidf_order, fdic_order)


# =============================================================================
# PART III: Rank and return (rule based)
# =============================================================================
# Create a rule based rank and return function

def rank(term):

    # get results from search
    results = search(term)
    # get metrics
    search_words = results[1] # search words found in dictionary
    num_search_words = len(results[1]) # number of search words found in dictionary
    num_score = results[2] # number of search words (as in dict) in each doc (in descending order)
    per_score = results[3] # percentage of search words (as in dict) in each doc (in descending order)
    tfscore = results[4] # sum of tfidf of search words in each doc (in ascending order)
    order_score = results[5] # fidc order

    # list of documents in order of relevance
    final_candidates = []


    ## no search term(s) not found
    if num_search_words == 0:
        print('Search term(s) not found')


    ## single term searched (as in dict): return the following 5 scores
    if num_search_words == 1:
        num_score_list = [l[0] for l in num_score] # document numbers
        num_score_list = num_score_list[:3] # take max 3 documents from num_score
        num_score_list.append(per_score[0][0]) # add the best percentage score
        num_score_list.append(tfscore[0][0]) # add the best tf score
        final_candidates = list(set(num_score_list)) # remove duplicate document numbers


    ## more than one search word (and found in dictionary)
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
                
        # rule 1a: add first document with 100% match of search_words
        # TODO check other scores highest ranking
        if per_score[0][1] == 1: # percentage score of first document number with 100% score
            final_candidates.append(per_score[0][0]) # document number

        # rule2: add max 4 other words with order_score greater than 1 (if not yet in final_candidates)
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
        df_results.loc[index, 'Title'] = df.title[results]
        df_results.loc[index, 'Paper_id'] = df.paper_id[results]
        df_results.loc[index, 'Document_no'] = results
        df_results.loc[index, 'Authors'] = df.authors[results]
        df_results.loc[index, 'Abstract'] = df.abstract[results]
        search_results = search_sentence(results, ' '.join(search_words))
        df_results.loc[index, 'Sentences'] = search_results

        # All search words per document 
        df_results.loc[index, 'Search_words'] = [word for word in search_words for sub_list in search_results if word in sub_list]

    return final_candidates, df_results


# =============================================================================
# PART IV: Function search sentence
# =============================================================================
# TODO rewrite without break
def search_sentence(doc_number, searchsentence):
    searchsentence = searchsentence.lower()
    search_words = searchsentence.split(' ')
    sentence_index = []
    for sentence in sentences[doc_number]:
        # if any(search_word in search_words for search_word in sentence.lower()):
        for search_word in search_words:
            if search_word in sentence.lower():
                sentence_index.append(sentence)
                break
    return sentence_index
            

# =============================================================================
# PART V: Function print result (ranked papers)
# =============================================================================
def print_ranked_papers(ranked_result, top_n=3, show_sentences=True, show_wordcloud=True):

    # Print top n result
   for index in range(top_n):    
       
        if pd.isnull(ranked_result.Title[index]):
            print('\n\nRESULT {}:'. format(index+1), 'Title not available')
        else: 
            print('\n\nRESULT {}:'. format(index+1), ranked_result.Title[index]) # Print Result from 1 and not 0
        print('\nII Number of search words in paper:', dict(Counter(ranked_result.Search_words.iloc[index])))       
        print('\nI Paper ID:', ranked_result.Paper_id[index], '(Document no: {})'. format(ranked_result.Document_no[index]))
        if pd.isnull(ranked_result.Abstract[index]):
            print('\nIII Authors:', 'Authors not available')
        else:
            print('\nIII Authors:', ranked_result.Authors[index])
        print('\n')
        if pd.isnull(ranked_result.Abstract[index]):
            print('Abstract not available')
        else: 
            print(ranked_result.Abstract[index])
            
        # join all sentences and seperate by a return
        text_sentences = '\n'.join(ranked_result.Sentences[index])
        # Spit in seperate words 
        text_sentences_split = text_sentences.split()

        search_words = list(set(ranked_result.Search_words[index]))

        # Generate cloud word
        if show_wordcloud == True:
            text_sentences_split = text_sentences.split()
            wordcloud = WordCloud()
            img = wordcloud.generate_from_text(' '.join(text_sentences_split))
            img.to_file('wordcloud{}.jpeg'.format(index))
         
            # plot word cloud       
            # plt.imshow(img)
            # plt.close()
            # image = Image.open('wordcloud.jpeg')
            # image.show()
            
            # %pylab inline
            img = mpimg.imread('wordcloud{}.jpeg'.format(index))
            imgplot = plt.imshow(img)
            plt.show()

        # Show sentences with search words in color
        if show_sentences == True:

            ## highlight search words
            # Set color
            # red = "\033[31m"
            green = "\033[32m"
            # blue = "\033[34m"
            reset = "\033[39m"
     
            # wrap search words in color
            for word in search_words:
                idxs = [i for i, x in enumerate(text_sentences_split) if x == word]
                for i in idxs:
                    text_sentences_split[i] = green + text_sentences_split[i] + reset
            # join the list back into a string and print
            text_sentences_colored = ' '.join(text_sentences_split)

            print('IV Sentences in paper containing search words:\n')
            print(text_sentences_colored)
    

# =============================================================================
# PART VI: Examples
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
papers, rank_result = rank('Full genome phylogenetic analysis')
papers, rank_result = rank('Sustainable risk reduction strategies')
rank('Full genome phylogenetic analysis')
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


# tests with df_biorxiv only0
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
# Example
# =============================================================================
search_example = 'Full-genome phylogenetic analysis'

papers, rank_result = rank(search_example)

# Print final candidates
print('Ranked papers (document numbers):', papers)

# Print results
print_ranked_papers(rank_result, top_n=3, show_sentences=True, show_wordcloud=True)


# =============================================================================
# CASE 1: Real-time tracking of whole genomes
# =============================================================================
papers, rank_result = rank('Real-time tracking of whole genomes and a mechanism for coordinating the rapid dissemination of that information to inform the development of diagnostics and therapeutics and to track variations of the virus over time.')

# Print final candidates
print('Ranked papers (document numbers):', papers)

# Print results
print_ranked_papers(rank_result, top_n=3, show_sentences=True, show_wordcloud=True)


# =============================================================================
# CASE 2: Access to geographic and temporal diverse sample sets
# =============================================================================
papers, rank_result = rank('Access to geographic and temporal diverse sample sets to understand geographic distribution and genomic differences, and determine whether there is more than one strain in circulation. Multi-lateral agreements such as the Nagoya Protocol could be leveraged.')

# Print final candidates
print('Ranked papers (document numbers):', papers)

# Print results
print_ranked_papers(rank_result, top_n=3, show_sentences=True, show_wordcloud=True)


# =============================================================================
# CASE 3: Evidence that livestock could be infected 
# =============================================================================
papers, rank_result = rank('Evidence that livestock could be infected (e.g., field surveillance, genetic sequencing, receptor binding) and serve as a reservoir after the epidemic appears to be over.')

# Print final candidates
print('Ranked papers (document numbers):', papers)

# Print results
print_ranked_papers(rank_result, top_n=3, show_sentences=True, show_wordcloud=True)


# =============================================================================
# CASE 4: Animal host(s) and any evidence of continued spill-over to humans
# =============================================================================
papers, rank_result = rank('Animal host(s) and any evidence of continued spill-over to humans')

# Print final candidates
print('Ranked papers (document numbers):', papers)

# Print results
print_ranked_papers(rank_result, top_n=3, show_sentences=True, show_wordcloud=True)


# =============================================================================
# CASE 5: Socioeconomic and behavioral risk factors for this spill-over
# =============================================================================
papers, rank_result = rank('Socioeconomic and behavioral risk factors for this spill-over')

# Print final candidates
print('Ranked papers (document numbers):', papers)

# Print results
print_ranked_papers(rank_result, top_n=1, show_sentences=True, show_wordcloud=True)


# =============================================================================
# CASE 6: Sustainable risk reduction strategies
# =============================================================================
papers, rank_result = rank('Full-genome phylogenetic analysis')

# Print final candidates
print('Ranked papers (document numbers):', papers)

# Print results
print_ranked_papers(rank_result, top_n=3, show_sentences=True, show_wordcloud=True)


