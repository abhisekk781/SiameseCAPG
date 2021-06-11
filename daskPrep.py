

# =========================================================================================================================
#   File Name           :   daskPrep.py
# -------------------------------------------------------------------------------------------------------------------------
#   Purpose             :   Purpose of this script is to apply different preprocessing functions and provide the output text 
#   Author              :   Abhisek Kumar
#   Co-Author           :   
#   Creation Date       :   13-November-2020
#   History             :
# -------------------------------------------------------------------------------------------------------------------------
#   Date            | Author                        | Co-Author                                          | Remark
#   28-August-2020    | Abhisek Kumar                                         | Initial Release
# =========================================================================================================================
# =========================================================================================================================
# Import required Module / Packages and start the cluster

#!pip install en_core_web_sm
#!pip install bs4
#!pip install nltk
#nltk.download('all')
#Required additional files : (contractions.py, stopwords.txt)
# -------------------------------------------------------------------------------------------------------------------------

from dask.distributed import Client
import pandas as pd
import numpy as np
import dask.dataframe as dd
import nltk
import en_core_web_sm
from bs4 import BeautifulSoup
import re
import unicodedata
from contractions import CONTRACTION_MAP
from nltk.tokenize.toktok import ToktokTokenizer
tokenizer = ToktokTokenizer()
stopword_list = nltk.corpus.stopwords.words('english')
from nltk.corpus import words
engwords = words.words()
nlp = en_core_web_sm.load()
import warnings
warnings.filterwarnings('ignore')
#Load the custom stopword file
# -------------------------------------------------------------------------------------------------------------------------
custok = []
with open('stopwords.txt', 'r') as f:
    for word in f:
        word = word.split('\n')
        custok.append(word[0])
# -------------------------------------------------------------------------------------------------------------------------
#Define the Schema
dtypes ={
    'Description': np.str
}


###########################################################################################################################
# Author        : Abhisek Kumar                                                                                        
# Functionality : Pre-Processing  removal different procedures                                                         
###########################################################################################################################

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

def custom_stopwords(text, custok=custok):
    tokens = tokenizer.tokenize(text)
    tokens = [token.strip() for token in tokens]
    filtered_custokens = [token for token in tokens if token not in custok]
    filtered_text = ' '.join(filtered_custokens) 
    return filtered_text

def get_keywords(text, eng_words = engwords):
    tokens = tokenizer.tokenize(text)
    eng_tokens = [token for token in tokens if token in eng_words]
    eng_text = ' '.join(eng_tokens)    
    return eng_text

def simple_porter_stemming(text):
    ps = nltk.porter.PorterStemmer()
    text = ' '.join([ps.stem(word) for word in text.split()])   
    return text

def lemmatize_text(text):
    text = nlp(text)
    text = ' '.join([word.lemma_ if word.lemma_ != '-PRON-' else word.text for word in text])
    return text

def remove_repeated_words(text):
    tokens = tokenizer.tokenize(text)
    tokens = [token.strip() for token in tokens]
    seen = set()
    seen_add = seen.add

    def add(x):
        seen_add(x)  
        return x
    text = ' '.join(add(i) for i in tokens if i not in seen)
    return text

###########################################################################################################################
# Author        : Abhisek Kumar                                                                                        
# Functionality : Map all functions in particular order                                                         
###########################################################################################################################

def clean_text(pDf,pcol):
    pDf = pDf.dropna(subset = [pcol])
    pDf[pcol] = pDf[pcol].apply(lambda s: str(s))
    pDf['Sample'] = pDf[pcol].map(lambda s: s.lower()).map(strip_html_tags).map(remove_special_characters).map(
        expand_contractions).map(remove_stopwords).map(custom_stopwords).map(get_keywords).map(remove_repeated_words)
    return pDf

###########################################################################################################################
# Author        : Abhisek Kumar                                                                                        
# Functionality : Load dataframe into dask, partition it and call all the preprocess functions                                                         
###########################################################################################################################
def preprocess(pData,pcol):
    try:
        #cluster = LocalCluster()
        client = Client(processes=False)
        print('Successfully started cluster...')
        dask_dataframe = dd.from_pandas(pData, npartitions=8)
        result = dask_dataframe.map_partitions(clean_text, pcol)
        pDfNew = result.compute()
        print('Pre-Processing Done')
        client.close()
    except Exception as e:
        client.close()
        print('failed')
    return (0, pDfNew)
