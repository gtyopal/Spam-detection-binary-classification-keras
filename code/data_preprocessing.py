# coding: utf-8
# encoding=utf8
import sys
import codecs
import random
import re
import nltk
from collections import defaultdict
from nltk.tokenize import word_tokenize
import pandas as pd
import numpy as np
import re
import string
import nltk
from sklearn.utils import shuffle
from bs4 import BeautifulSoup
from nltk.stem import LancasterStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer


def clean_str(string):
    #remove url """
    string = str(string)
    string = re.sub(r'(http|ftp|https):\/\/[\w\-_]+(\.[\w\-_]+)+([\w\-\.,@?^=%&amp;:/~\+#]*[\w\-\@?^=%&amp;/~\+#])?',
                    ' spamurl ', string)
    # remove email """
    string = re.sub(r'([\w-]+(\.[\w-]+)*@[\w-]+(\.[\w-]+)+)', ' email ', string)
    # remove phone numbers """
    string = re.sub(r'[\@\+\*].?[014789][0-9\+\-\.\~\(\) ]+.{6,}', ' phone ', string)
    # remove digits """
    string = re.sub(r'[0-9\.\%]+', ' digit ', string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.encode('utf-8').strip().lower()


def strip_html(string):
    soup = BeautifulSoup(string, "html.parser")
    string = soup.get_text()
    r = re.compile(r'<[^>]+>', re.S)
    string = r.sub('', string)
    string = re.sub(r'&(nbsp;)', ' ', string)
    string = re.sub(r'<[^>]+', '', string)
    string = re.sub('\&lt[;]', ' ', string)
    string = re.sub('\&gt[;]', ' ', string)
    return string

def denoise_text(string):
    string = clean_str(string)
    string = strip_html(string)
    words = word_tokenize(string)
    string = " ".join(words)
    if not string.strip():
        string = " "
    return string

def load_data_from_disk():
    # Load dataset from file
    df_ham = pd.read_csv(ham_path, sep=",", encoding='latin-1', low_memory=False)
    df_spam = pd.read_csv(spam_path, sep=",", encoding='latin-1', low_memory=False)

    # remove all Unnamed Columns form the CSV File
    df_ham.drop(list(df_ham.filter(regex='Unnamed')), axis=1, inplace=True)
    df_spam.drop(list(df_spam.filter(regex='Unnamed')), axis=1, inplace=True)

    # concatenate both SUBJECT and BODY
    df_ham['message'] = df_ham.SUBJECT.str.cat(df_ham.BODY)
    df_spam['message'] = df_spam.SUBJECT.str.cat(df_spam.BODY)

    # drop the columns SUBJECT from both ham and spam files
    df_ham.drop(['SUBJECT', 'BODY'], axis=1, inplace=True)
    df_spam.drop(['SUBJECT', 'BODY'], axis=1, inplace=True)

    # replace na with null
    df_ham.fillna(value='null', inplace=True)
    df_spam.fillna(value='null', inplace=True)

    # adding labels
    df_ham['label'] = 'ham'
    df_spam['label'] = 'spam'

    # merge both data frames HAM and SPAM into One.
    df = df_ham.append(df_spam, ignore_index=True)
    df = shuffle(df)

    # very important otherwise df[0]->(message) length and df[1]->(label) length are mismatched
    df = df[pd.notnull(df['message'])]

    # drop all NaN rows from the data frame
    df.dropna()

    # drop all rows with NULL value from data frame
    df = df[~df['message'].str.contains('null')]
    df['label'] = df['label'].map({'ham': 0, 'spam': 1})
    df['message'] = df['message'].apply((lambda x: denoise_text(x)))

    # save clean all data
    df.to_csv("../data/all_clean.csv", sep="\t", index=False, columns=["label", "message"])
    return df


if __name__ == '__main__' :
    spam_path = "../data/spam.csv"
    ham_path = "../data/ham.csv"
    load_data_from_disk()