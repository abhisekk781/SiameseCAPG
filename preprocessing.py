import nltk
import re
#import spacy
#from nltk import word_tokenize
from bs4 import BeautifulSoup
import unicodedata
from contractions import CONTRACTION_MAP
from nltk.corpus import wordnet
#import collections
from nltk.tokenize.toktok import ToktokTokenizer
from bs4 import BeautifulSoup
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import en_core_web_sm
import pandas as pd
import sys

from nltk.corpus import words
engwords = words.words()



###########################################################################################################################
# Author        : Tapas  Mohanty                                                                                        
# Functionality : Pre-Processing  removal different procedures                                                         
###########################################################################################################################

tokenizer = ToktokTokenizer()
stopword_list = nltk.corpus.stopwords.words('english')
# nlp = spacy.load('en', parse=True, tag=True, entity=True)
nlp = en_core_web_sm.load()
# nlp_vec = spacy.load('en_vectors_web_lg', parse=True, tag=True, entity=True)



def strip_html_tags(text):
    soup = BeautifulSoup(text, "html.parser")
    if bool(soup.find()):
        [s.extract() for s in soup(['iframe', 'script'])]
        stripped_text = soup.get_text()
        stripped_text = re.sub(r'[\r|\n|\r\n]+', '\n', stripped_text)
        stripped_text = re.sub(r'''(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'".,<>?«»“”‘’]))''', " ", stripped_text)    
    else:
        stripped_text = text
    # print('Strip html tags completed')
    return stripped_text


def simple_porter_stemming(text):
    ps = nltk.porter.PorterStemmer()
    text = ' '.join([ps.stem(word) for word in text.split()])   
    # print('Stemming completed')
    return text



def lemmatize_text(text):
    text = nlp(text)
    text = ' '.join([word.lemma_ if word.lemma_ != '-PRON-' else word.text for word in text])
    # print('Lemmatiation completed')
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
    # print('remove repeated words completed')
    return text
    
    
def remove_repeated_characters(text):
    repeat_pattern = re.compile(r'(\w*)(\w)\2(\w*)')
    match_substitution = r'\1\2\3'
    tokens = tokenizer.tokenize(text)
    tokens = [token.strip() for token in tokens]
    def replace(old_word):
        if wordnet.synsets(old_word):
            return old_word
        new_word = repeat_pattern.sub(match_substitution, old_word)
        return replace(new_word) if new_word != old_word else new_word
            
    correct_tokens = [replace(word) for word in tokens]
    # print('remove repeated characters')
    return correct_tokens


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
    # print('expand contractions completed')
    return expanded_text


def remove_accented_chars(text):
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')
    # print('removal accented chars')
    return text


def remove_special_characters(text, remove_digits=False):
    pattern = r'[^a-zA-Z0-9\s]|\[|\]' if not remove_digits else r'[^a-zA-Z\s]|\[|\]'
    text = re.sub(pattern, '', text)
    # print('removal special characters completed')
    return text


def remove_stopwords(text, is_lower_case=False, stopwords = stopword_list):
    tokens = tokenizer.tokenize(text)
    tokens = [token.strip() for token in tokens]
    if is_lower_case:
        filtered_tokens = [token for token in tokens if token not in stopwords]
    else:
        filtered_tokens = [token for token in tokens if token.lower() not in stopwords]
    filtered_text = ' '.join(filtered_tokens)    
    # print('removal stopwords completed')
    return filtered_text

def custom_stopwords(text, custok):
    tokens = tokenizer.tokenize(text)
    tokens = [token.strip() for token in tokens]
    filtered_custokens = [token for token in tokens if token not in custok]
    filtered_text = ' '.join(filtered_custokens) 
    # print('removal custom stopwords completed')
    return filtered_text
    
def get_keywords(text, eng_words = engwords):
    tokens = tokenizer.tokenize(text)
    eng_tokens = [token for token in tokens if token in eng_words]
    eng_text = ' '.join(eng_tokens)    
    # print('removal of non-english keywords completed')
    return eng_text
    
def col_keyword(pData, pTktDesc, column):
    try:
        pData['combined'] = pData[column].apply(lambda row: ' '.join(row.values.astype(str)), axis=1)
        pData[pTktDesc] = ([' '.join(set(a.split(' ')).difference(set(b.split(' ')))) for a, b in zip(pData[pTktDesc], pData['combined'])])
        del pData['combined']
    except Exception as e:
        print('*** Error[001]: ocurred while combining column')
    return pData


def normalize_corpus(corpus, html_stripping= True, contraction_expansion= True,
                     accented_char_removal= True, text_lower_case= True, 
                     text_stemming= False, text_lemmatization= True, 
                     special_char_removal= True, remove_digits= True,
                     stopword_removal= True, ewords = True,
                     custm_stpwrds= True, stopwords=stopword_list,
                     remove_rptd_wrds= True, eng_words = engwords):
    
    normalized_corpus = []
    # normalize each document in the corpus
    
    custok = []
    with open('stopwords.txt', 'r') as f:
       for word in f:
            word = word.split('\n')
            custok.append(word[0])
    print('--> preprocessing started')        
    for index, doc in enumerate(corpus):
        # print(index)        
        try: 
            # strip HTML
            if html_stripping:
                doc = strip_html_tags(doc)
        except Exception as e:
            print('*** Error[002]: ocurred in html_stripping on row no: ', index)

        try: 
            # remove extra newlines
            doc = doc.translate(doc.maketrans("\n\t\r", "   "))
        except Exception as e:
            print('*** Error[003] ocurred on row no: ', index)
        
        try:        
            # remove accented characters
            if accented_char_removal:
                doc = remove_accented_chars(doc)
        except Exception as e:
            print('*** Error[004]: ocurred in accented_char_removal on row no: ', index)
        
        try:        
            # expand contractions    
            if contraction_expansion:
                doc = expand_contractions(doc)
        except Exception as e:
            print('*** Error[005] ocurred in contraction_expansion on row no: ', index)
            
        try:    
            # lemmatize text
            if text_lemmatization:
                doc = lemmatize_text(doc)
        except Exception as e:
            print('*** Error[006]: ocurred in text_lemmatization on row no: ', index)
            
        try:    
            # stem text
            if text_stemming and not text_lemmatization:
                doc = simple_porter_stemming(doc)
        except Exception as e:
            print('*** Error[007]: ocurred in text_stemming on row no: ', index)
        
        try:
            # remove special characters and\or digits    
            if special_char_removal:
                # insert spaces between special characters to isolate them    
                special_char_pattern = re.compile(r'([{.(-)!}])')
                doc = special_char_pattern.sub(" \\1 ", doc)
                doc = remove_special_characters(doc, remove_digits=remove_digits)  
        except Exception as e:
            print('*** Error[008] ocurred in special_char_removal on row no: ', index)
            
        try:    
            # remove extra whitespace
            doc = re.sub(' +', ' ', doc)
        except Exception as e:
            print('*** Error[009]: ocurred on row no: ', index)
            
        try:
            # lowercase the text    
            if text_lower_case:
                doc = doc.lower()
        except Exception as e:
            print('*** Error[010]: ocurred in text_lower_case on row no: ', index)
            
        try:            
            # remove stopwords
            if stopword_removal:
                doc = remove_stopwords(doc, is_lower_case=text_lower_case, stopwords = stopwords)
        except Exception as e:
            print('*** Error[011]: ocurred in stopword_removal on row no: ', index)

        try:                
            #Remove non-english keywords
            if eng_words:
                doc = get_keywords(doc, eng_words = eng_words)
        except Exception as e:
            print('*** Error[012]: ocurred in ewords on row no: ', index)

        try:            
            #Remove custom keywords
            if custm_stpwrds:
                doc = custom_stopwords(doc, custok)

            # remove extra whitespace
            doc = re.sub(' +', ' ', doc)
            doc = doc.strip()
        except Exception as e:
            print('*** Error[013]: ocurred in custm_stpwrds on row no: ', index)

        try:            
            #Remove repeated words
            if remove_rptd_wrds:
                doc = remove_repeated_words(doc)
            # remove extra whitespace
            doc = re.sub(' +', ' ', doc)
            doc = doc.strip()
        except Exception as e:
            print('*** Error[014]: ocurred in remove_rptd_wrds on row no: ', index)
            
        normalized_corpus.append(doc)
    print('--> preprocessing completed')        
    return normalized_corpus
    


def preprocess(pData, pTktDesc, pCol):
    pData.columns = pData.columns.str.lower()   
    pData = pData.applymap(str)
    
    pData = col_keyword(pData, pTktDesc, pCol)
    pData = pData.dropna(subset = [pTktDesc])    
    try:
        norm_corpus = normalize_corpus(corpus=pData[pTktDesc], html_stripping=True, contraction_expansion=True, 
                                      accented_char_removal=True, text_lower_case=True, text_lemmatization=True, 
                                      text_stemming=False, special_char_removal=True, remove_digits=True,
                                      custm_stpwrds= True,stopword_removal=True, ewords = True, stopwords=stopword_list,
                                      eng_words = engwords)
    except Exception as e:
        print('Error ocurred due to template')
        return(-1)
                                  
    pData['Sample'] = norm_corpus

    return (0,pData)   
    




