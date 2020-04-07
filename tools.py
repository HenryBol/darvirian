
# =============================================================================
# Import datasets
# =============================================================================

class Import(object):

    def importing_datasets(self):
        """Importing the dataset and setting the id to index"""
        import pandas as pd     
        self.df = pd.read_csv('output/new_Multiclass.csv', sep = ',')
        # Index: set Ids to index if the IDs are unique
        if self.df['id'].is_unique:
            self.df.set_index('id', inplace=True)
        self.df.shape # (917, 76)
        self.df.keys()
        return(self.df)

        # Load pickle file plot_data
        pickle_in = open("Data/output/plot_data.pkl","rb")
        plot_data = pickle.load(pickle_in) 

        ## Load pickle file df_dfidf
        pickle_in = open("Data/output/df_tfidf.pkl","rb")
        df_dfidf = pickle.load(pickle_in) 

        # Load pickle file worddic
        pickle_in = open("Data/output/worddic_all_200406.pkl","rb")
        worddic = pickle.load(pickle_in) 
            
        return plot_data, df_dfidf, worddic


# =============================================================================
# Search
# =============================================================================
class Search(object):
    
    def __init__(self):
        # dummy variabele zodat er een blok is
        self.test = 0
              
    def search(self, searchsentence):
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
                x = [] # per word: document(s) that a word appeards
                y = [] # documents with more than one search word
                for record in [worddic[z] for z in words]:
                    for index in record:
                          x.append(index[0])
                for i in x:               
                    if x.count(i) > 1:
                        y.append(i)
                y = list(set(y))
    
                # dictionary of documents and all positions per word (for docs with more than one search word in it)
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
    
                # metric: fdic order
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
