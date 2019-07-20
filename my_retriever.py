import math
import numpy as np
class Retrieve:
    # Create new Retrieve object storing index and termWeighting scheme
    def __init__(self,index, termWeighting):
        self.index = index
        self.amountOfDocs = self.getAmountOfDocs()
        # if tw is tfidf then get the inverse doc frequency
        if (termWeighting == "tfidf"):
            self.idf = self.getInverseDocFreq()
        self.termWeighting = termWeighting
        # Retrieve the document vector for specified term weighting
        self.documentVector = self.getDocumentVector(termWeighting)
        
    # Method performing retrieval for specified query
    def forQuery(self, query):
        if (self.termWeighting == "binary"):
            weightedDocs = self.getBinarySimilarities(query)
        if (self.termWeighting == "tf"):
            weightedDocs = self.getTfSimilarities(query)
        if (self.termWeighting == "tfidf"):
            weightedDocs = self.getTfidfSimilarities(query)
            
        # Sort similarity scores and return top 10
        topDocs = np.argsort(weightedDocs)[::-1][:10]
        for i in range (10):
            topDocs[i] += 1
        return topDocs
    
    # Calculate the total amount of documents
    def getAmountOfDocs(self):
        docId = []
        # Compute set of docids 
        for term in self.index:
            for docid in self.index[term]:
                if docid not in docId:
                    docId.append(docid)
        # Return the size of docids list
        return len(docId)
    
    # Calculate the inverse document frequency of each term
    def getInverseDocFreq(self):
        idf = {}
        for term in self.index:
            idf[term] = math.log10(self.amountOfDocs/len(self.index[term]))
        return idf
    
    def getDocumentVector(self, tw):
        docVect = {}
        docSizes = [0] * self.amountOfDocs
        
        if (tw == "binary"):   
            # Create a doc vector for binary term weight
            for word in self.index:
                for doc in self.index[word]:
                    if doc in docVect:
                        docVect[doc][word] = 1
                    else:
                        docVect[doc] = {word:1}
            # Loop through doc vect and work out doc sizes
            for doc in docVect:
                value = 0
                for word in docVect[doc]:
                    value+=1
                docSizes[doc-1] = math.sqrt(value)
                        
        if (tw == "tf"):
            # Create a doc vector for term frequency term weight
            for word in self.index:
                for doc in self.index[word]:
                    if doc in docVect:
                        docVect[doc][word] = self.index[word][doc]
                    else:
                        docVect[doc] = {word:self.index[word][doc]}
            # Loop through doc vect and work out doc sizes          
            for doc in docVect:
                value = 0
                for word in docVect[doc]:
                    # Square each term frequency 
                    value+=math.pow(docVect[doc][word], 2)
                # Square root the final value of the document
                docSizes[doc-1] = math.sqrt(value)
                
        if (tw == "tfidf"):
            idf = self.idf
            # Create a doc vector for term frequency term weight
            for word in self.index:
                for doc in self.index[word]:
                    if doc in docVect:
                        # Multiple term frequency by the idf value of the word
                        docVect[doc][word] = self.index[word][doc]*idf[word]
                    else:
                        docVect[doc] = {word:self.index[word][doc]*idf[word]}
            # Loop through doc vect and work out doc sizes  
            for doc in docVect:
                docSum = 0
                for word in docVect[doc]:
                     # Square each term frequency 
                    docSum+=math.pow(docVect[doc][word], 2)
                # Square root the final value of the document
                docSizes[doc-1] = math.sqrt(docSum)
        # Return document vector and document sizes
        return docVect, docSizes
                
    def getBinarySimilarities(self, query):
        # retrieve the document vector and doc size
        dv, docSizes = self.documentVector
        # initialise a list of 0s 
        similarities = [0] * self.amountOfDocs
        # get the qd product for docs
        for word in query:
            if word in self.index:
                for doc in self.index[word]:
                    similarities[doc-1] += 1
        # loop through every doc similarity score and divide it by docs size
        for i in range(self.amountOfDocs):
            similarities[i] /= docSizes[i]
        return similarities
    
    def getTfSimilarities(self, query):
        dv, docSizes = self.documentVector
        similarities = [0] * self.amountOfDocs
        # get the qd product for docs
        for word in query:
            if word in self.index:
                for doc in self.index[word]:
                    similarities[doc-1] += dv[doc][word] * query[word]
        # loop through every doc similarity score and divide it by docs size
        for i in range(self.amountOfDocs):
            similarities[i] /= docSizes[i]
        return similarities
    
    def getTfidfSimilarities(self, query):
        dv, docSizes = self.documentVector
        similarities = [0] * self.amountOfDocs
        weightedQuery = {}
        # Workout the tf.idf scores for query
        for word in query:
            if word in self.idf:
                weightedQuery[word] = query[word] * self.idf[word]
        # get the qd product for docs
        for word in query:
            if word in self.index:
                for doc in self.index[word]:
                    similarities[doc-1] += dv[doc][word] * weightedQuery[word]
        # loop through every doc similarity score and divide it by docs size
        for i in range(self.amountOfDocs):
            similarities[i] /= docSizes[i]
        return similarities