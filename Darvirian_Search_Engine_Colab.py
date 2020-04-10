from collections import Counter
import re
import pickle
import pandas as pd


df_biorxiv = pd.read_csv('Data/CSV/biorxiv_clean.csv')
df_clean_comm_use = pd.read_csv('Data/CSV/clean_comm_use.csv')
df_clean_noncomm_use = pd.read_csv('Data/CSV/clean_noncomm_use.csv')
df_clean_pmc = pd.read_csv('Data/CSV/clean_pmc.csv')


df = df_biorxiv.append(df_clean_comm_use).reset_index(drop=True)
df = df.append(df_clean_noncomm_use).reset_index(drop=True)
df = df.append(df_clean_pmc).reset_index(drop=True)


pickle_in = open('Data/output/sentences_200407.pkl', 'rb')
sentences = pickle.load(pickle_in)


pickle_in = open('Data/output/plot_data_200407.pkl', 'rb')
plot_data = pickle.load(pickle_in)


pickle_in = open('Data/output/worddic_all_200407.pkl', 'rb')
worddic = pickle.load(pickle_in)



def search(searchsentence):

    searchsentence = searchsentence.lower()

    words = searchsentence.split(' ')
    words = [re.sub(r'[^a-zA-Z.]', '', str(w)) for w in words]

    enddic = {}
    idfdic = {}
    closedic = {}


    words = [word for word in words if word in worddic.keys()]
    numwords = len(words)
    
    for word in words:

        for indpos in worddic[word]:

            index = indpos[0]
            amount = len(indpos[1])
            idfscore = indpos[2]

            if index in enddic.keys():
                enddic[index] += amount
                idfdic[index] += idfscore

            else:
                enddic[index] = amount
                idfdic[index] = idfscore
    fullcount_order = sorted(enddic.items(), key=lambda x: x[1], reverse=True)
    fullidf_order = sorted(idfdic.items(), key=lambda x: x[1], reverse=True)

    alloptions = {k: worddic.get(k) for k in words}
    comboindex = [item[0] for worddex in alloptions.values() for item in worddex]
    combocount = Counter(comboindex)
    for key in combocount:
        combocount[key] = combocount[key] / numwords
    combocount_order = sorted(combocount.items(), key=lambda x: x[1], reverse=True)


    if len(words) > 1:
        x = [index[0] for record in [worddic[z] for z in words] for index in record]
        y = sorted(list(set([i for i in x if x.count(i) > 1])))


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
                    fdic_order = sorted(fdic.items(), key=lambda x: x[1], reverse=True)
                while x == 0:
                    firstlist = seqlist
                    x = x + 1 
    else:
        fdic_order = 0


    return(searchsentence, words, fullcount_order, combocount_order, fullidf_order, fdic_order)




def rank(term):


    results = search(term)

    num_search_words = len(results[1]) 
    num_score = results[2] 
    per_score = results[3] 
    tfscore = results[4] 
    order_score = results[5] 


    final_candidates = []


    if num_search_words == 0:
        print('Search term(s) not found')


    if num_search_words == 1:
        num_score_list = [l[0] for l in num_score] 
        num_score_list = num_score_list[:3] 
        num_score_list.append(per_score[0][0])
        num_score_list.append(tfscore[0][0]) 
        final_candidates = list(set(num_score_list))



    if num_search_words > 1:

        first_candidates = []


        for candidates in order_score:
            if candidates[1] >= 1:
                first_candidates.append(candidates[0])

        second_candidates = []

        for match_candidates in per_score:

            if match_candidates[1] == 1:
                second_candidates.append(match_candidates[0])

        for match_candidates in first_candidates:
            if match_candidates in second_candidates:
                final_candidates.append(match_candidates)


        t3_order = first_candidates[0:3]
        for each in t3_order:
            if each not in final_candidates:
                final_candidates.insert(len(final_candidates), each)

        final_candidates.insert(len(final_candidates), tfscore[0][0])
        final_candidates.insert(len(final_candidates), tfscore[1][0])

        t3_per = second_candidates[0:3]
        for each in t3_per:
            if each not in final_candidates:
                final_candidates.insert(len(final_candidates), each)

        othertops = [num_score[0][0], per_score[0][0], tfscore[0][0], order_score[0][0]]
        for top in othertops:
            if top not in final_candidates:
                final_candidates.insert(len(final_candidates), top)



    print('\nFound search words:', results[1])
    print('Ranked papers (document numbers):', final_candidates)

    df_results = pd.DataFrame(columns=['Title', 'Paper_id', 'Document_no', 'Authors', 'Abstract', 'Sentences'])
    for index, results in enumerate(final_candidates):
        # if index < 5:
        df_results.loc[index+1, 'Title'] = df.title[results]
        df_results.loc[index+1, 'Paper_id'] = df.paper_id[results]
        df_results.loc[index+1, 'Document_no'] = results
        df_results.loc[index+1, 'Authors'] = df.authors[results]
        df_results.loc[index+1, 'Abstract'] = df.abstract[results]
        search_results = search_sentence(results, term)
        df_results.loc[index+1, 'Sentences'] = search_results

    return final_candidates, df_results

def search_sentence(doc_number, search_term):
    sentence_index = []
    search_list = search_term.split()
    for sentence in sentences[doc_number]:
        for search_word in search_list:
            if search_word.lower() in sentence.lower():
                sentence_index.append(sentence) 
    return sentence_index



search_case_7 = search('Sustainable risk reduction strategies')
search('Sustainable risk reduction strategies')
rank('Sustainable risk reduction strategies')

papers, rank_result = rank('Sustainable risk reduction strategies')
rank_result.to_csv('Data/output/rank_result_0200410.csv')
