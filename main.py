import pandas as pd
import re
from many_stop_words import get_stop_words
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from itertools import chain
import pickle
import nltk
import os.path
nltk.download('stopwords')

from nltk.classify import NaiveBayesClassifier, accuracy
stop_words = list(get_stop_words('en'))
nltk_words = list(stopwords.words('english'))
stop_words.extend(nltk_words)
import threading

# the function use to clean tweets such as remove links and anything else 0-9 a-z A-Z
def clean_content(content):
    content = content.lower()
    return ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", " ", content).split())

# this function use stop words from tweet such as is, the, and, etc
def clean_stopwords(list):
    cleaned = ""
    for word in list:
        word = word.lower()
        if word not in stopwords.words("english"):
            cleaned = cleaned + " " + word

    return cleaned.lstrip()

#-----------------------------------------------------------------------------------------------------------------------

#uncomment below code to train model again and generate confusion matrix
print("Training Model")
df = pd.read_csv("trainset.csv",encoding = "ISO-8859-1")# open trainset file
df = df.dropna()
df = df.sample(frac=1)# shuffle file
training_data = []
print("preparing data")
count=0
for index,row in df.iterrows():
    if row['label'] == 'love' or row['label'] == 'joy' or row['label'] == 'fear' or row['label'] == 'anger' or row['label'] == 'surprise' or row['label'] == 'sadness':
        if len(row['label'].strip()) != 0 and len(row['sentence'].strip()) != 0:
            row['sentence'] = clean_content(row['sentence'])# get tweet/sentence/content
            row['sentence'] = clean_stopwords(row['sentence'].split())# remove stop words
            training_data.append([row['sentence'], row['label']])# append tweet and its associated true label to array
            count = count + 1

print("feature set")
tokens = set(chain(*[word_tokenize(i[0].lower()) for i in training_data]))# create tokens of training data

# generate feature set
feature_set = [({i: (i in word_tokenize(sentence.lower())) for i in tokens}, tag) for sentence, tag in
               training_data]
print("naive bayes classifier")
# split data into train:test
size = int(len(feature_set) * 0.1)
train_set, test_set = feature_set[size:], feature_set[:size]


classifier = NaiveBayesClassifier.train(train_set)# train mode
# save trained model
f = open('my_classifier.pickle', 'wb')
pickle.dump(classifier, f)
f.close()
# display informative features computed by naive bayes algo
classifier.show_most_informative_features()
print("Training Completed")
print("Accuracy: ",nltk.classify.accuracy(classifier, test_set))# measure accuracy
# compute confusion matrix
print("Confusion Matrix")
test_result = []
gold_result = []

for i in range(len(test_set)):
    test_result.append(classifier.classify(test_set[i][0]))
    gold_result.append(test_set[i][1])

CM = nltk.ConfusionMatrix(gold_result, test_result)
print(CM)

#-----------------------------------------------------------------------------------------------------------------------
# load trained model from directory
f = open('my_classifier.pickle', 'rb')
classifier = pickle.load(f)
f.close()

# run test file
print("Analysing test file")
output = {'love':0,'fear':0,'anger':0,'sadness':0,'surprise':0,'joy':0}
process = 0
input = pd.read_csv("test.csv")
size = input.shape[0]
#input = input[0:10000]

for index,row in input.iterrows():
    tweet = str(row['cleaned_tweet'])
    if len(tweet) > 0:
        sentence = clean_content(tweet)# cleant tweet
        sentence = clean_stopwords(tweet.split())# remove stop words
        featurized_tweet =  {i:(i in word_tokenize(tweet.lower())) for i in sentence}#convert into feature array
        tag = classifier.classify(featurized_tweet)# predict its emotion
        output[tag] = output[tag] + 1# count weight of each emotion

for tag in output:
    output[tag] = output[tag]/size#get average of each emotion in document
print(output)
