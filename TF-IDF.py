# Search engine


# =============================================================================
# Module TF-IDF (term frequency inverse document frequency)
# Goal: determine the tf-idf for each word in a dataset 'plottest'
# Output: worddic

# Input plottest: list of lists
# 1st level: documents
# 2nd level: 

# Output worddic: dictionary
# KEY: word 
# VALUES: list of doc indexes where the word occurs plus per doc: word position(s) and tfidf-score

# =============================================================================

import pandas as pd

df_biorxiv = pd.read_csv('biorxiv_clean.csv')
df = biorixv.copy()
df  = df[:100] # Set to subest for testign purposes



from collections import defaultdict
worddic = defaultdict(list)

# Loop version
for doc in plottest:
    for word in set(doc): # set provides unique words in doc 
        word = str(word)
        index = plottest.index(doc)
        positions = [index for index, w in enumerate(doc) if w == word]
        idfs = tfidf(word,doc,plottest)
        worddic[word].append([index,positions,idfs])


# Comprehension version
import time
start = time.time()
[worddic[word].append([plottest.index(doc), [index for index, w in enumerate(doc) if w == word], tfidf(word,doc,plottest)]) for doc in plottest for word in set(doc)]
end = time.time()
print(end - start) # duration 2.0 hours for biorxiv
# TD-IDF processing direct: no impact
# plottest_length = len(plottest)
# [worddic[word].append([plottest.index(doc), [index for index, w in enumerate(doc) if w == word], (doc.count(word) / len(doc)) / np.log(plottest_length / sum(1 for doc in plottest if word in doc))]) for doc in plottest for word in set(doc)]

    