import json
import datetime
import os.path
import numpy as np
from evaluator import Evaluator
from metrics import Metrics
from filewriter import FileWriter
from math import sqrt
from nltk.tokenize import RegexpTokenizer
from stop_words import get_stop_words
from nltk.stem.porter import PorterStemmer
from gensim import corpora, models

test_set_path = "testSet"
training_set_path = "trainingSet"
topics = 100
passes = 1
test_set_limit = 200
threshold_start = 1
threshold_end = 10

thresholds = []
metrics_obj_list = []

fw = FileWriter()

for i in range(threshold_start, threshold_end+1):
    thresholds.append(i)
    metrics_obj_list.append(Metrics())

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
training_set = json.load(open(training_set_path))["documents"]
training_set_texts = [doc["abstractText"] for doc in training_set]
tlog("Training set read.");

if test_set_limit !=-1:
    test_set = json.load(open(test_set_path))["documents"][0:test_set_limit]
else:
    test_set = json.load(open(test_set_path))["documents"]

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

    fw.writeQueryResults(results[0:thresholds[-1]], i)

    for k in range(0, len(thresholds)):
        threshold = thresholds[k]

        eval.query([y for (x,y) in results[0:threshold]], test_set[i])
        eval.calculate()

        metrics_obj_list[k].updateConfusionMatrix(eval)
        metrics_obj_list[k].updateMacroAverages(eval)


for obj in metrics_obj_list:
    obj.calculate(len(test_set))

fw.writeToFiles(metrics_obj_list, thresholds)

for i in range(0, len(thresholds)):
    print("Threshold: "+str(thresholds[i]) + " Recall: " + str(metrics_obj_list[i].ma_recall) + " " + str(metrics_obj_list[i].mi_recall))

tlog("Done.")

end_time = getTime()

print("Start time: " + start_time)
print("End time: " + end_time)
