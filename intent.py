import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.corpus import words
engwords = words.words()
from sparse_dot_topn import awesome_cossim_topn
from scipy.sparse import csr_matrix
from typing import List
import yake
import traceback
import sys
import string 
import en_core_web_sm
from bs4 import BeautifulSoup
import re
import unicodedata
from contractions import CONTRACTION_MAP
from nltk.tokenize.toktok import ToktokTokenizer
tokenizer = ToktokTokenizer()
stopword_list = nltk.corpus.stopwords.words('english')
nlp = en_core_web_sm.load()
import warnings
warnings.filterwarnings('ignore')
#Load the custom stopword file
# -------------------------------------------------------------------------------------------------------------------------
custok = []
with open('./stopwords.txt', 'r') as f:
    for word in f:
        word = word.split('\n')
        custok.append(word[0])

def strip_html_tags(text):
    soup = BeautifulSoup(text, "html.parser")
    if bool(soup.find()):
        [s.extract() for s in soup(['iframe', 'script'])]
        stripped_text = soup.get_text()
        stripped_text = re.sub(r'[\r|\n|\r\n]+', '\n', stripped_text)
        stripped_text = re.sub(r'''(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'".,<>?«»“”‘’]))''', " ", stripped_text)    
    else:
        stripped_text = text
    return stripped_text

def remove_accented_chars(text):
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')
    return text

def remove_special_characters(text, remove_digits=False):
    pattern = r'[^a-zA-Z0-9\s]|\[|\]' if not remove_digits else r'[^a-zA-Z\s]|\[|\]'
    text = re.sub(pattern, '', text)
    # print('removal special characters completed')
    return text

def expand_contractions(text, contraction_mapping=CONTRACTION_MAP):
    
    contractions_pattern = re.compile('({})'.format('|'.join(contraction_mapping.keys())), 
                                      flags=re.IGNORECASE|re.DOTALL)
    def expand_match(contraction):
        match = contraction.group(0)
        first_char = match[0]
        expanded_contraction = contraction_mapping.get(match)\
                                if contraction_mapping.get(match)\
                                else contraction_mapping.get(match.lower())                       
        expanded_contraction = first_char+expanded_contraction[1:]
        return expanded_contraction
        
    expanded_text = contractions_pattern.sub(expand_match, text)
    expanded_text = re.sub("'", "", expanded_text)
    return expanded_text

def remove_stopwords(text, is_lower_case=False, stopwords = stopword_list):
    tokens = tokenizer.tokenize(text)
    tokens = [token.strip() for token in tokens]
    if is_lower_case:
        filtered_tokens = [token for token in tokens if token not in stopwords]
    else:
        filtered_tokens = [token for token in tokens if token.lower() not in stopwords]
    filtered_text = ' '.join(filtered_tokens)    
    return filtered_text

def custom_stopwords(text, stopList, custok=custok):
    tokens = tokenizer.tokenize(text)
    tokens = [token.strip() for token in tokens]
    if stopList is not None:
        custok.extend(stopList)
    filtered_custokens = [token for token in tokens if token not in custok]
    filtered_text = ' '.join(filtered_custokens) 
    return filtered_text

def lemmatize_text(text):
    text = nlp(text)
    text = ' '.join([word.lemma_ if word.lemma_ != '-PRON-' else word.text for word in text])
    return text

def simple_porter_stemming(text):
    ps = nltk.porter.PorterStemmer()
    tokens = [ps.stem(word) for word in text.split()]   
    return tokens

def remove_repeated_words(text):
    tokens = text.split(' ')
    tokens = list(set(tokens))
    text = ' '.join(tokens)
    return text


def process_text(sentence):
    '''Make text lowercase, remove text in square brackets,remove links,remove punctuation
    and remove words containing numbers.'''
    #lower case 
    sentence = sentence.lower()
    # square brackets
    sentence = re.sub('\[', ' ', sentence)
    sentence = re.sub(']', ' ', sentence)
    # curly brackets
    sentence = re.sub('\{', ' ', sentence)
    sentence = re.sub('}', ' ', sentence)
    # square brackets
#     sentence = re.sub('\[.*?\]', '', sentence)  
    # hyperlinks
    sentence = re.sub('https?://\S+|www\.\S+', '', sentence)
    sentence = re.sub('<.*?>+', '', sentence)
    # remove punctuation
    sentence = re.sub('[%s]' % re.escape(string.punctuation), '', sentence)
    sentence = re.sub('\n', '', sentence)
    # remove year
    sentence = re.sub(r"\b(19[40][0-9]|20[0-1][0-9]|2020)\b",'',sentence)
    # remove month names
    mp = r"(\b\d{1,2}\D{0,3})?\b(?:Jan|jan(?:uary)?|Feb|feb(?:ruary)?|Mar|mar(?:ch)?|Apr|mar(?:il)?|May|may|Jun|jun(?:e)?|Jul|jul(?:y)?|Aug|aug(?:ust)?|Sep|sep(?:tember)?|Oct|oct(?:ober)?|(Nov|nov|Dec|dec)(?:ember)?)\D?(\d{1,2}(st|nd|rd|th)?)?(([,.\-\/])\D?)?((19[7-9]\d|20\d{2})|\d{2})*"
    sentence = re.sub(mp, '', sentence)
    # remove words containing numbers
    sentence = re.sub('\w*\d\w*', '', sentence)
    # remove numbers 
    sentence = re.sub(r'[0-9]+','',sentence)
    sentence = " ".join(sentence.split())
    return sentence

def get_keywords(text, eng_words = engwords):
    tokens = tokenizer.tokenize(text)
    eng_tokens = [token for token in tokens if token in eng_words]
    eng_text = ' '.join(eng_tokens)    
    return eng_text

def preprocess(pDf,pcol,stopList):
    pDf = pDf.dropna(subset = [pcol])
    pDf[pcol] = pDf[pcol].apply(lambda s: str(s))
    pDf['Sample'] = pDf[pcol].map(lambda s: s.lower()).map(strip_html_tags).map(remove_special_characters).map(expand_contractions).map(remove_stopwords).map(lambda s: custom_stopwords(s,stopList)).map(process_text).map(lemmatize_text).map(remove_repeated_words).map(get_keywords)
    pDf = pDf.dropna(subset = ['Sample'])
    return pDf


def filterTopicNames(text, stopwords = stopword_list, stem = False, lemma = True):
    text = text.lower()
    text = re.sub('[^a-zA-Z0-9\s]|\[|\]', ' ', text)
    text = re.sub(' +', ' ', text)
    text = text.strip()
    tokens = text.split(' ')
    filtered_tokens = [token for token in tokens if token.lower() not in stopwords]
    
    if lemma is True:
        filtered_text = nlp(' '.join(filtered_tokens))
        filtered_tokens = [word.lemma_ if word.lemma_ != '-PRON-' else word.text for word in filtered_text]
        
    if stem is True:
        ps = nltk.porter.PorterStemmer()
        filtered_tokens = [ps.stem(word) for word in filtered_tokens]
        
    filtered_tokens = list(set(filtered_tokens))
    return '__'.join(filtered_tokens)

def fillmissing(df, col_lst, fill_lst):
    return df.fillna(dict(zip(col_lst, fill_lst)))


#Use logic and assign high, medium, low recommendation
def recommendLogic(sample, intent):
    ps = nltk.porter.PorterStemmer()
    sample_arr = [ps.stem(word) for word in sample.split()]
    intent_arr = [ps.stem(word) for word in intent.split('_')]
    matchings = 0.0
    if len(sample_arr) >= len(intent_arr):
        for word in intent_arr:
            if word in sample_arr:
                matchings+= 1.0
        # Logic for high recommendation
        if matchings/float(len(intent_arr)) >= 0.9:
        #Penalty term
            if np.mean([matchings/float(len(intent_arr)), float(len(intent_arr))/float(len(sample_arr))]) >= 0.6:
                return 'high'
            else: return 'medium'
        elif matchings/float(len(intent_arr)) >= 0.7:
            return 'medium'
        elif matchings == 0.0:
            return 'zero'
        elif matchings/float(len(intent_arr)) < 0.7:
            return 'low'
        
    else:
        for word in sample_arr:
            if word in intent_arr:
                matchings+= 1.0
        # Logic for high recommendation
        if matchings/float(len(sample_arr)) >= 0.9: 
        #Penalty term
            if np.mean([matchings/float(len(sample_arr)), float(len(sample_arr))/float(len(intent_arr))]) >= 0.6:
                return 'high'
            else: return 'medium'
        elif matchings/float(len(sample_arr)) >= 0.7:
            return 'medium'
        elif matchings == 0.0:
            return 'zero'
        elif matchings/float(len(sample_arr)) < 0.7:
            return 'low'




#Assign intent visibility
def visibilityLogic(pData, intentCol):
    pData['count'] = [1 if (x == 'high' or x == 'medium') else 0 for x in pData['intent_recommendation']]
    unique_intent = list(set(pData[intentCol].values))
    intent_dict = {}
    for intent in unique_intent:
        x = pData.groupby(intentCol).get_group(intent)['count'].sum()
        y = len(pData.groupby(intentCol).get_group(intent)['count'].values)
        #Visibility ranges
        if y <= 2:
            if x == 2: intent_dict[intent] = 'good'
            if x >= 1: intent_dict[intent] = 'ok'
            else: intent_dict[intent] = 'bad'
        elif y <= 4:
            if x == 4: intent_dict[intent] = 'good'
            if x >= 2: intent_dict[intent] = 'ok'
            else: intent_dict[intent] = 'bad'
        elif y <= 5:
            if x == 5: intent_dict[intent] = 'good'
            if x >= 3: intent_dict[intent] = 'ok'
            else: intent_dict[intent] = 'bad'
        else:
            if x >= int(np.round(0.9*y)):
                intent_dict[intent] = 'good'
            elif x >= int(np.round(0.5*y)):
                intent_dict[intent] = 'ok'
            elif x < int(np.round(0.5*y)):
                intent_dict[intent] = 'bad'
    pData['intent_visibility'] = [intent_dict[i] for i in pData[intentCol]]
    pData = pData.drop(columns = ['count'])
    return pData


# Function for cosine similarity and ngram generation
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



def getIntent(pTopicData, pQueryData, pKeyCol, pDescCol, pTopicDesc, pTktno, nbest = 1):
    try:
        #Get the input parameters
        listOfDataFrames = []
        #pTopicData = pTopicData.dropna(subset = [pTopicDesc])
        pQueryData = pQueryData.dropna(subset = [pDescCol])
        to_list = pQueryData[pDescCol].values.astype('U').tolist()
        Train_CC = pQueryData[pKeyCol].values.astype('U').tolist()
        keyNames = pTopicData[pTopicDesc].values.astype('U').tolist()
        tktno = pTopicData[pTktno].values
        nbest = int(nbest)
        for item, n in zip(keyNames, tktno):
            from_list = [item]
            if len(to_list) < nbest:
                continue
            vectorizer = TfidfVectorizer(min_df=1, analyzer=_create_ngrams).fit(to_list + from_list)
            X = vectorizer.transform(to_list)
            Y = vectorizer.transform(from_list)
            to_vector = csr_matrix(X)
            from_vector = csr_matrix(Y)
            matches = cosine_similarity(Y, X, from_list, to_list, Train_CC, nbest)
            tempDF = pd.DataFrame()
            tempDF[pKeyCol] = matches.iloc[:,nbest+1:2*nbest+1].values.tolist()[-1]
            tempDF[pTktno] = nbest*[n]    
            tempDF['similarity_score'] = [float(x) for x in matches.iloc[:,2*nbest+1:].values.tolist()[-1]]
            listOfDataFrames.append(tempDF)
        result = pd.concat(listOfDataFrames)
        merged = pd.merge(result, pTopicData, how = 'inner', on = pTktno)
        return merged
    except Exception as e:
        print(traceback.format_exc(), e)
        return (-1)


def kw_extractor(text, language = "en", max_ngram_size = 3,  deduplication_threshold = 0.3, numOfKeywords = 25, features = None):
    custom_kw_extractor = yake.KeywordExtractor(lan=language, n=max_ngram_size, dedupLim=deduplication_threshold, top=numOfKeywords, features=features)
    keywords = custom_kw_extractor.extract_keywords(text)
    return ' '.join(list(set(" ".join([x[0] for x in keywords]).split())))







