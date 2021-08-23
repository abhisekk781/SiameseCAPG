import findspark
findspark.init()
findspark.find()
import pyspark
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.types import *
import time
import re
import nltk
from bs4 import BeautifulSoup
import unicodedata
from nltk.tokenize.toktok import ToktokTokenizer
tokenizer = ToktokTokenizer()
stopword_list = nltk.corpus.stopwords.words('english')
from nltk.corpus import words
engwords = words.words()
from nltk.corpus import wordnet
import traceback


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


custok = []
with open('stopwords.txt', 'r') as f:
    for word in f:
        word = word.split('\n')
        custok.append(word[0])

def custom_stopwords(text, custok = custok):
    tokens = tokenizer.tokenize(text)
    tokens = [token.strip() for token in tokens]
    filtered_custokens = [token for token in tokens if token not in custok]
    filtered_texts = ' '.join(filtered_custokens) 
    # print('removal custom stopwords completed')
    return filtered_texts



def get_keywords(text, eng_words = engwords):
    tokens = tokenizer.tokenize(text)
    eng_tokens = [token for token in tokens if token in eng_words]
    eng_text = ' '.join(eng_tokens)    
    # print('removal of non-english keywords completed')
    return eng_text


def simple_porter_stemming(text):
    ps = nltk.porter.PorterStemmer()
    text = ' '.join([ps.stem(word) for word in text.split()])   
    # print('Stemming completed')
    return text



import en_core_web_sm
nlp = en_core_web_sm.load()
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




def text_preprocessing(pData, pCol):
    try:
        df = pd.DataFrame()
        df[pCol] = pData[pCol].values
        config = pyspark.SparkConf().setAll([('spark.executor.memory', '8g'), 
                                             ('spark.executor.cores', '3'), 
                                             ('spark.cores.max', '3'), 
                                             ('spark.driver.memory','8g')])


        spark = SparkSession.builder.config(conf=config).getOrCreate()
        schema = StructType([StructField(pCol, StringType(), True)])
        sdf = spark.createDataFrame(df, schema=schema)

        sentenceDescriptionRDD = sdf.select(pCol).rdd.flatMap(lambda x: x)
        htmlTagsRemovedRDD = sentenceDescriptionRDD.map(strip_html_tags)
        acntCharRemovedRDD = htmlTagsRemovedRDD.map(remove_accented_chars)
        specCharRemovedRDD = acntCharRemovedRDD.map(remove_special_characters)
        lowerCaseRDD = specCharRemovedRDD.map(lambda x : x.lower())
        stopWordsRemovedRDD = lowerCaseRDD.map(remove_stopwords)
        customStopWordsRemovedRDD = stopWordsRemovedRDD.map(custom_stopwords)
        nonEngWordsRemovedRDD = customStopWordsRemovedRDD.map(get_keywords)
        porterStemmingRDD = nonEngWordsRemovedRDD.map(simple_porter_stemming)
        lemmaRDD = porterStemmingRDD.map(lemmatize_text)
        repeatedWordsRemovedRDD = lemmaRDD.map(remove_repeated_words)

        start_time = time.time()
        finalTextProcessed = repeatedWordsRemovedRDD.collect()
        Tokens = list(map(lambda s : remove_repeated_characters(s),finalTextProcessed))
        time_elapsed = time.time() - start_time
        print('############TIME ELAPSED###########----',time_elapsed,'----')
    except Exception:
        print(traceback.format_exc())
    
    return Tokens