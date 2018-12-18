# coding:utf-8
import argparse
import os, sys
from classifier.text_classifier import TextClassifier


""" Command or IDE run """
#Usage: python predict_cmd.py --conf config/spam.json

parse = argparse.ArgumentParser()
parse.add_argument("--conf", help="config file", required=True)
args = parse.parse_args()
conf = args.conf

# init model
model = TextClassifier(conf_path=conf, ispredict=1)

while True:
    sentence = input("Enter sentence, Ctrl+C to exit:")
    y_pred = model.predict(sentence)
    if y_pred[0][0] > y_pred[0][1]:
        spam_class = "Spam"

    else:
        spam_class = "Ham"

    print(sentence)
    print("Spam Classification : ", spam_class)
    print(y_pred)
    print("[Spam prob , ", "Ham prob]")
