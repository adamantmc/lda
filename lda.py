import json
import datetime
import os.path
import numpy as np
from evaluator import Evaluator
from math import sqrt
from nltk.tokenize import RegexpTokenizer
from stop_words import get_stop_words
from nltk.stem.porter import PorterStemmer
from gensim import corpora, models

topics = 100
passes = 10
test_set_limit = 10

#Tokenizer
tokenizer = RegexpTokenizer(r'\w+')
#Stop word list
en_stop = get_stop_words('en')
#Porter Stemmer
p_stemmer = PorterStemmer()

def getTime():
    return str(datetime.datetime.time(datetime.datetime.now()))

def tlog(msg):
    print("["+getTime()+"] "+msg)

def cossim(v1, v2):
    v1_mag = np.linalg.norm(v1)
    v2_mag = np.linalg.norm(v2)
    product = np.dot(v1,v2)
    
    cosine_sim = product / (v1_mag * v2_mag)
    return cosine_sim

def processTexts(texts):
    processed_texts = []
    for text in texts:
        # clean and tokenize document string
        raw = text.lower()
        tokens = tokenizer.tokenize(raw)

        # remove stop words from tokens
        stopped_tokens = [i for i in tokens if not i in en_stop]

        # stem tokens
        stemmed_tokens = [p_stemmer.stem(i) for i in stopped_tokens]

        # add tokens to list
        processed_texts.append(stemmed_tokens)
    return processed_texts

def buildDictionary(texts):
    tlog("Building dictionary.")
    # turn our tokenized documents into a id <-> term dictionary
    dictionary = corpora.Dictionary(texts)
    tlog("Dictionary built.")
    return dictionary

def buildCorpus(texts, dictionary):
    tlog("Building corpus.")
    # convert tokenized documents into a document-id matrix
    corpus = [dictionary.doc2bow(text) for text in texts]
    tlog("Corpus built.")
    return corpus

def generateLDA(corpus, dictionary, topic_num, pass_num):
    tlog("Generating LDA Model.")
    # generate LDA model
    ldamodel = models.LdaMulticore(corpus, num_topics = topic_num, id2word = dictionary, passes = pass_num, workers = 3)
    tlog("LDA Model generated.")
    return ldamodel

start_time = getTime()

#Reading both sets
training_set = json.load(open("trainingSet"))["documents"]
training_set_texts = [doc["abstractText"] for doc in training_set]
tlog("Training set read.");

if test_set_limit !=-1: 
    test_set = json.load(open("testSet"))["documents"][0:test_set_limit]
else: 
    test_set = json.load(open("testSet"))["documents"]

test_set_texts = [doc["abstractText"] for doc in test_set]
tlog("Test set read.")

print()

#Processing both sets - tokenization, stop-word removal, stemming
tlog("Processing training set.")
training_processed_texts = processTexts(training_set_texts)
tlog("Training set processed.")

tlog("Processing test set.")
test_processed_texts = processTexts(test_set_texts)
tlog("Test set processed.")

#Build a dictionary and a corpus on training set
train_dict = buildDictionary(training_processed_texts)
train_corpus = buildCorpus(training_processed_texts, train_dict)

#Read lda from disk, if not available build it and save it
if os.path.isfile("./ldamodel"):
    print("["+getTime()+"] Loading saved LDA model from disk.")
    lda = models.ldamodel.LdaModel.load(fname="./ldamodel")
else:
    lda = generateLDA(train_corpus, train_dict, topics, passes)
    lda.save("./ldamodel")

#Build corpus on test set
test_set_corpus = buildCorpus(test_processed_texts, train_dict)

#Query and evaluate results
eval = Evaluator(training_set)
threshold = 10

tp = 0
tn = 0
fp = 0
fn = 0
ma_accuracy = 0
ma_precision = 0
ma_recall = 0
ma_f1score = 0
mi_accuracy = 0
mi_precision = 0
mi_recall = 0
mi_f1score = 0

tlog("Creating training set topic list.")
train_topic_list = []
for i in range(0, len(training_set)):
    train_topic_list.append([x[1] for x in lda.get_document_topics(train_corpus[i], minimum_probability=0)])
tlog("Topic list done.")

for i in range(0, len(test_set)):
    query_doc = test_set_corpus[i]
    query_doc_topics = [x[1] for x in lda.get_document_topics(query_doc, minimum_probability=0)]

    tlog("Calculating similarities.")

    results = []
    for j in range(0,len(training_set)):
        cos_sim = cossim(train_topic_list[j], query_doc_topics)
        results.append((cos_sim, training_set[j]))

    tlog("Sorting by similarity score.")
    results.sort(key=lambda tup: tup[0], reverse=True)

    for (x,y) in results[0:threshold]:
        print(y["title"])

    eval.query([y for (x,y) in results[0:threshold]], test_set[i])
    eval.calculate()

    tp += eval.getTp()
    tn += eval.getTn()
    fp += eval.getFp()
    fn += eval.getFn()

    ma_precision += eval.getPrecision()
    ma_recall += eval.getRecall()
    ma_accuracy += eval.getAccuracy()
    ma_f1score += eval.getF1Score()

mi_accuracy = (tp+tn)/(tp+tn+fp+fn)
mi_precision = tp/(tp+fp)
mi_recall = tp/(tp+fn)
mi_f1score = 2*mi_precision*mi_recall/(mi_precision+mi_recall)

ma_accuracy = ma_accuracy / len(test_set)
ma_precision = ma_precision / len(test_set)
ma_recall = ma_recall / len(test_set)
ma_f1score = ma_f1score / len(test_set)

print("Results:")
print("================================")
print("Micro-average:")
print("Accuracy: "+str(mi_accuracy))
print("Precision: "+str(mi_precision))
print("Recall: "+str(mi_recall))
print("F1Score: "+str(mi_f1score))
print("================================")
print("Macro-average:")
print("Accuracy: "+str(ma_accuracy))
print("Precision: "+str(ma_precision))
print("Recall: "+str(ma_recall))
print("F1Score: "+str(ma_f1score))
print("================================")
print("TP: "+str(tp))
print("TN: "+str(tn))
print("FP: "+str(fp))
print("FN: "+str(fn))
print("================================")

tlog("Done.")

end_time = getTime()

print("Start time: " + start_time)
print("End time: " + end_time)
