import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import re
from nltk.corpus import stopwords
from sparse_dot_topn import awesome_cossim_topn
from scipy.sparse import csr_matrix
from typing import List
import psycopg2
import traceback
import sys

def cosine_similarity(from_vector: np.ndarray,
                      to_vector: np.ndarray,
                      from_list: List[str],
                      to_list: List[str],
                      to_key_list: List[str],
                      nbest,
                      min_similarity: float = 0) -> pd.DataFrame:
    
    if nbest != None:
        if int(nbest) >  len(to_list):
            raise ValueError('best choice must be less than to_list')
    else:
        nbest = int(1)

    if isinstance(to_vector, np.ndarray):
        to_vector = csr_matrix(to_vector)
    if isinstance(from_vector, np.ndarray):
        from_vector = csr_matrix(from_vector)

    # There is a bug with awesome_cossim_topn that when to_vector and from_vector
    # have the same shape, setting topn to 1 does not work. Apparently, you need
    # to it at least to 2 for it to work

    if int(nbest) <= 1:
        similarity_matrix = awesome_cossim_topn(from_vector, to_vector.T, 2, min_similarity)
    elif int(nbest) > 1:
        similarity_matrix = awesome_cossim_topn(from_vector, to_vector.T, nbest, min_similarity)

    if from_list == to_list:
        similarity_matrix = similarity_matrix.tolil()
        similarity_matrix.setdiag(0.)
        similarity_matrix = similarity_matrix.tocsr()

    if int(nbest) <= 1:
        indices = np.array(similarity_matrix.argmax(axis=1).T).flatten()
        similarity = similarity_matrix.max(axis=1).toarray().T.flatten()
    elif int(nbest) > 1:
        similarity = np.flip(np.take_along_axis(similarity_matrix.toarray(), np.argsort(similarity_matrix.toarray(), axis =1), axis=1) [:,-int(nbest):], axis = 1)
        indices = np.flip(np.argsort(np.array(similarity_matrix.toarray()), axis =1)[:,-int(nbest):], axis = 1)
            
    
    if int(nbest) <= 1:
        matches = [to_list[idx] for idx in indices.flatten()]
        key_matches = [to_key_list[idx] for idx in indices.flatten()]
        matches = pd.DataFrame(np.vstack((from_list, matches, key_matches, similarity)).T, columns=["From", "To", "Key", "Similarity"])
        matches.Similarity = matches.Similarity.astype(float)
        matches.loc[matches.Similarity < 0.001, "To"] = None
        matches.loc[matches.Similarity < 0.001, "Key"] = None
    else:
        matches = [np.array([to_list[idx] for idx in l]) for l in indices] ##In progress
        key_matches = [np.array([to_key_list[idx] for idx in l]) for l in indices] ##In progress
        column = []
        column.append("To")
        for i in range(int(nbest) - 1):
            column.append("BestMatch" + "__" + str(i+1))
        column.append("Key")
        for j in range(int(nbest) - 1):
            column.append("Key" + "__" + str(j+1))
        column.append("Similarity")
        for j in range(int(nbest) - 1):
            column.append("Similarity" + "__" + str(j+1))
            
        matches = pd.concat([pd.DataFrame({'From' : from_list}), pd.DataFrame(np.hstack((matches, key_matches, similarity)), columns= column)], axis =1)
        matches.Similarity = matches.Similarity.astype(float)
        matches.loc[matches.Similarity < 0.001, "To"] = None
        matches.loc[matches.Similarity < 0.001, "Key"] = None
        for i in range(int(nbest) - 1):
            matches.loc[matches.Similarity < 0.001, "BestMatch" + "__" + str(i+1)] = None
            matches.loc[matches.Similarity < 0.001, "Key" + "__" + str(i+1)] = None
        
    return matches


def _create_ngrams(string: str) -> List[str]:
    n_gram_range=(3, 3)
    string = _clean_string(string)
    result = []
    for n in range(n_gram_range[0], n_gram_range[1]+1):
        ngrams = zip(*[string[i:] for i in range(n)])
        ngrams = [''.join(ngram) for ngram in ngrams if ' ' not in ngram]
        result.extend(ngrams)
    return result


def _clean_string(string: str) -> str:
    """ Only keep alphanumerical characters """
    string = re.sub(r'[^A-Za-z0-9 ]+', '', string.lower())
    string = re.sub('\s+', ' ', string).strip()
    return string


def connectDB(pdatabase, pUser, pPassword, pHost, pPort):
    database = psycopg2.connect (database = pdatabase, user = pUser, password = pPassword, host = pHost, port = pPort)
    return(database)


def getInputDataFrame(pDatabase, pAsgCol, pAppCol, pTopicCol):
    cursor = pDatabase.cursor()
    try:
        query = """SELECT %s, %s, %s FROM history"""
        cursor.execute(query,(pAsgCol, pAppCol, pTopicCol))
        names = [pAsgCol, pAppCol, pTopicCol]
        rows = cursor.fetchall()
        pData = pd.DataFrame(rows, columns = names)
        cursor.close()
        return pData
    except Exception as e:
        cursor.close()
        print('*** ERROR[0001]: getInputDataFrame: ', sys.exc_info()[0],str(e))
        return(-1)
    
    
def getQueryDataFrame(pDatabase, pAsgCol, pAppCol, pKeyCol, pDescCol):
    cursor = pDatabase.cursor()
    try:
        query = """SELECT %s, %s, %s, %s FROM ticket"""
        cursor.execute(query,(pKeyCol, pDescCol, pAsgCol, pAppCol))
        names = [pKeyCol, pDescCol, pAsgCol, pAppCol]
        rows = cursor.fetchall()
        pData = pd.DataFrame(rows, columns = names)
        cursor.close()
        return pData
    except Exception as e:
        cursor.close()
        print('*** ERROR[0002]: getQueryDataFrame: ', sys.exc_info()[0],str(e))
        return(-1)
    
    
    
def loadRevisedTopics(pDatabase, pData, pAsgCol, pAppCol, pKeyCol, pTopicCol):
    cursor = pDatabase.cursor()
    totalRecordsProcessed = 0
    try:
        for i in range(len(pData)):
                try:
                    #UPDATE
                    queryU = """ UPDATE  ticket 
                            SET     Incident_Key = %s, application_name = %s, Topic = %s, assignment_group = %                                                       
                            WHERE   Incident_Key = %s """
                    
                    valuesU = (str(pData[pKeyCol].values[i]), str(pData[pAppCol].values[i]), str(pData[pTopicCol].values[i]), str(pData[pAsgCol].values[i]), str(pData[pKeyCol].values[i]))
                    cursor.execute(queryU,valuesU)
                    #INSERT
                    query = """INSERT INTO ticket (Incident_Key, application_name, Topic, assignment_group) 
                                                     SELECT  %s, %s, %s, %s
                                                     WHERE NOT EXISTS (  SELECT  'x' 
                                                                        FROM    ticket 
                                                                        WHERE   Incident_Key = %s)""" 

                    values = (str(pData[pKeyCol].values[i]), str(pData[pAppCol].values[i]), str(pData[pTopicCol].values[i]), str(pData[pAsgCol].values[i]), str(pData[pKeyCol].values[i]))
                    cursor.execute(query, values)
                    totalRecordsProcessed += 1
                    #Ongoing Commit after 50 rows
                    if( totalRecordsProcessed % 50 == 0 ):
                        pDatabase.commit()
                        print(totalRecordsProcessed, 'rows inserted...')

                except Exception as e:
                    print('Some error occured during row injection.')
                    print('*** ERROR[0003]: loadData:', sys.exc_info()[0], str(e))
                    print(traceback.format_exc())               
                    cursor.close()
                    return(-1)
        #Final Commit
        pDatabase.commit()
        cursor.close()
        print('Total', totalRecordsProcessed, ' rows inserted...')
        return (0)
    except Exception as e:
        cursor.close()
        print(traceback.format_exc())
        print('No rows inserted. Some error occured in database connection.')
        return(-1)
    
    
    
def getRelevantTopics(pDatabaseConn, pKeyCol, pAsgCol, pAppCol, pDescCol, pTopicCol, nbest = 5):
    try:
        #Get the input parameters
        pInputData = getInputDataFrame(pDatabaseConn, pAsgCol, pAppCol, pTopicCol):
        appNames = pInputData[pAppCol].tolist()
        topicNames = pInputData[pTopicCol].tolist()
        asgNames = pInputData[pAsgCol].tolist()
        listOfDataFrames = []
        #get the query dataframe
        pQueryData = getQueryDataFrame(pDatabase, pAsgCol, pAppCol, pKeyCol, pDescCol)
        #Initialize the top-n tickets
        nbest = int(nbest)
        #Remove Duplicates and make combined pairs for efficient input
        combinedList = [x+'|'+y+'|'+z for x, y, z in zip(asgNames, appNames, topicNames)]
        combDict = {}
        for item in combinedList:
            combDict[item] = 1
        for item in list(combDict.keys())[:]:
            tempList = item.split('|')
            inputAsgGroup = tempList[0]
            inputAppName = tempList[1]
            inputTopicName = tempList[2]
            from_list = [inputTopicName]
            to_list = pQueryData.groupby(pAsgCol).get_group(inputAsgGroup).groupby(pAppCol).get_group(inputAppName)[pDescCol].values.astype('U').tolist()
            Train_CC = pQueryData.groupby(pAsgCol).get_group(inputAsgGroup).groupby(pAppCol).get_group(inputAppName)[pKeyCol].values.astype('U').tolist()
            if len(to_list) < nbest:
                continue
            vectorizer = TfidfVectorizer(min_df=1, analyzer=_create_ngrams).fit(to_list + from_list)
            X = vectorizer.transform(to_list)
            Y = vectorizer.transform(from_list)
            to_vector = csr_matrix(X)
            from_vector = csr_matrix(Y)
            matches = cosine_similarity(Y, X, from_list, to_list, Train_CC, nbest)
            tempDF = pd.DataFrame()
            tempDF[pDescCol] = matches.iloc[:,1:nbest+1].values.tolist()[-1]
            tempDF[pKeyCol] = matches.iloc[:,nbest+1:2*nbest+1].values.tolist()[-1]
            tempDF['similarity_index_score'] = [float(x) for x in matches.iloc[:,2*nbest+1:].values.tolist()[-1]]
            tempDF[pAsgCol] = nbest*[inputAsgGroup]
            tempDF[pAppCol] = nbest*[inputAppName]
            tempDF[pTopicCol] = nbest*[inputTopicName]
            listOfDataFrames.append(tempDF)
            resultData = pd.concat(listOfDataFrames)
            loadRevisedTopics(pDatabaseConn, resultData, pAsgCol, pAppCol, pKeyCol, pTopicCol)
            return (0)
    except Exception as e:
        print(e)
        return (-1)