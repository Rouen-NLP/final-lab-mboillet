#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  8 15:31:00 2018

@author: boillmel
"""

import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix

#%% Store the text files and the contents

texts_names = dict()
texts_contents = dict()
text_category_couples = []
dir_names = os.listdir("./data/Tobacco3482-OCR")

for dir_name in dir_names:
    file_names = os.listdir("./data/Tobacco3482-OCR/"+dir_name)
    names = []
    contents = []
    for file_name in file_names:
        names.append(file_name)
        with open("./data/Tobacco3482-OCR/"+dir_name+"/"+file_name, 'r', encoding="utf8") as file:
            data = file.read()
            text_category_couples.append((data, dir_name))
            contents.append(data)
    # Create a dictionary with the categories as keys and an array of the
    # names of the text files as values.
    texts_names[dir_name] = names
    # Create a dictionary with the categories as keys and an array of the
    # contents of the text files as values.
    texts_contents[dir_name] = contents

#%% Plot the repartition of the texts by categories

plt.figure(figsize=(5, 5))
plt.bar(list(texts_names.keys()), [len(texts_names[category]) for category in texts_names.keys()])
plt.xlabel('Categorie')
plt.ylabel('Nombre de textes')
plt.title('Nombre de textes par categorie')
plt.show()

#%% Compute and plot the mean number of letters in the texts by categories

mean_lengths = dict()

for class_name in list(texts_names.keys()):
    sum_lengths = 0
    mean_length = 0
    for index, text in enumerate(texts_contents[class_name]):
        sum_lengths += len(text)
    mean_length = sum_lengths / (index + 1)
    mean_lengths[class_name] = mean_length

plt.figure()
plt.bar(list(mean_lengths.keys()), list(mean_lengths.values()))
plt.xlabel('Categorie')
plt.ylabel('Nombre moyen de lettres')
plt.title('Nombre moyen de lettres par categorie')
plt.show()

#%% Split the data, vectorize the texts vectors

X, y = zip(*text_category_couples)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)
X_train, X_dev, y_train, y_dev = train_test_split(X_train, y_train, test_size=(0.20 / (0.60 + 0.20)))

print('Train : {0:0} = {1:.2f} %'.format(len(X_train), (len(X_train)/len(X)*100)))
print('Dev : {0:0} = {1:.2f} %'.format(len(X_dev), (len(X_dev)/len(X)*100)))
print('Test : {0:0} = {1:.2f} %'.format(len(X_test), (len(X_test)/len(X)*100)))

#%% Create document vectors

vectorizer = CountVectorizer(max_features=2000)
vectorizer.fit(X_train)
X_train_counts = vectorizer.transform(X_train)
X_dev_counts = vectorizer.transform(X_dev)
X_test_counts = vectorizer.transform(X_test)

#%% Train a Naive Bayes classifier

clf = MultinomialNB()
clf.fit(X_train_counts, y_train)
train_predict = clf.predict(X_train_counts)
dev_predict = clf.predict(X_dev_counts)
test_predict = clf.predict(X_test_counts)

print("\nBag-of-word representation :")
print('Accuracy train set : {0:.2f} %'.format(len(train_predict[train_predict == y_train]) / len(y_train)*100))
print('Accuracy dev set : {0:.2f} %'.format(len(dev_predict[dev_predict == y_dev]) / len(y_dev)*100))
print('Accuracy test set : {0:.2f} %'.format(len(test_predict[test_predict == y_test]) / len(y_test)*100))

#%%

tf_transformer = TfidfTransformer().fit(X_train_counts)
X_train_tf = tf_transformer.transform(X_train_counts)
X_dev_tf = tf_transformer.transform(X_dev_counts)
X_test_tf = tf_transformer.transform(X_test_counts)

clf.fit(X_train_tf, y_train)
train_predict = clf.predict(X_train_tf)
dev_predict = clf.predict(X_dev_tf)
test_predict = clf.predict(X_test_tf)

print("\nTF-IDF representation")
print('Accuracy train set : {0:.2f} %'.format(len(train_predict[train_predict == y_train]) / len(y_train)*100))
print('Accuracy dev set : {0:.2f} %'.format(len(dev_predict[dev_predict == y_dev]) / len(y_dev)*100))
print('Accuracy test set : {0:.2f} %'.format(len(test_predict[test_predict == y_test]) / len(y_test)*100))

#%% Error analysis

print("\nScores per class Train")
print(classification_report(y_train, train_predict))
print("\nScores per class Dev")
print(classification_report(y_dev, dev_predict))
print("\nScores per class Test")
print(classification_report(y_test, test_predict))

print("\nConfusions Train")
plt.matshow(confusion_matrix(y_train, train_predict))
# TODO: change xticks with name categories
print("\nConfusions Dev")
plt.matshow(confusion_matrix(y_dev, dev_predict))
print("\nConfusions Test")
plt.matshow(confusion_matrix(y_test, test_predict))
