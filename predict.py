# coding:utf-8
import argparse
import os
from classifier.text_classifier import TextClassifier
import warnings
warnings.filterwarnings("ignore", message="numpy.dtype size changed")

""" Command or IDE run instruction:
if cmd = False, then run on IDE
if cmd = True, then run on Command
Command run Usage: python predict.py --conf config/spam.json """

cmd = False
conf = "config/spam.json"

if cmd:
    parse = argparse.ArgumentParser()
    parse.add_argument("--conf", help="config file", required=True)
    args = parse.parse_args()
    conf = args.conf

# init model
model = TextClassifier(conf_path=conf, ispredict=1)

# predict
text = "Hello darling how are you today? I would love to have a chat, why dont you tell me what you look like and what you are in to sexy?"

y_pred = model.predict(text)

if y_pred[0][0] > y_pred[0][1]:
    spam_class = "Spam"

else:
    spam_class = "Ham"

print(text)
print("Spam Classification : ", spam_class )
print(y_pred)
print("[Spam prob , ", "Ham prob]")
# print("Ham Ratio : ", abs(y_pred[0][1]) / abs(y_pred[0][0]-y_pred[0][1]))
