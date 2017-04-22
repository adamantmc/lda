class Evaluator:
    def __init__(self, all_docs):
        all_labels = set()
        for doc in all_docs:
            for label in doc["meshMajor"]:
                all_labels.add(label)
        self.all_label_count = len(all_labels)

    def query(self, retrieved_docs, query_doc):
        self.retrieved_docs = retrieved_docs
        self.query_doc = query_doc

    def calculate(self):
        self.tp = 0
        self.tn = 0
        self.fp = 0
        self.fn = 0

        self.average_doc_precision = 0
        self.average_doc_recall = 0
        self.average_doc_f1score = 0
        self.average_precision = 0

        retrieved_labels = set()
        prev_recall = 0

        for doc in self.retrieved_docs:
            relevant = 0

            for label in doc["meshMajor"]:
                retrieved_labels.add(label)
                if label in self.query_doc["meshMajor"]:
                    relevant += 1

            map_relevant = 0
            for label in retrieved_labels:
                if label in self.query_doc["meshMajor"]:
                    map_relevant += 1

            prec_at_i = map_relevant / len(retrieved_labels)
            rec_at_i = map_relevant / len(self.query_doc["meshMajor"])

            self.average_precision += prec_at_i*(rec_at_i - prev_recall)
            prev_recall = rec_at_i

            av_doc_prec = relevant / len(doc["meshMajor"])
            av_doc_rec = relevant / len(self.query_doc["meshMajor"])
            self.average_doc_precision += av_doc_prec
            self.average_doc_recall += av_doc_rec
            if av_doc_prec != 0 and av_doc_rec != 0:
                self.average_doc_f1score += 2*(av_doc_prec*av_doc_rec)/(av_doc_rec+av_doc_prec)

        for label in self.query_doc["meshMajor"]:
            if label in retrieved_labels:
                self.tp = self.tp + 1
                retrieved_labels.remove(label)
            else:
                self.fn = self.fn + 1

        self.average_doc_precision = self.average_doc_precision / len(self.retrieved_docs)
        self.average_doc_recall = self.average_doc_recall / len(self.retrieved_docs)
        self.average_doc_f1score = self.average_doc_f1score / len(self.retrieved_docs)
        self.fp = len(retrieved_labels)
        self.tn = self.all_label_count - self.fp - self.fn - self.tp
        self.combined_f1score = 2 * (self.average_doc_precision * self.getRecall())/(self.average_doc_precision + self.getRecall())

    def getTp(self):
        return self.tp
    def getTn(self):
        return self.tn
    def getFp(self):
        return self.fp
    def getFn(self):
        return self.fn

    def getAccuracy(self):
        return (self.tp + self.tn)/(self.tp + self.tn + self.fp + self.fn)
    def getPrecision(self):
        return self.tp / (self.tp + self.fp)
    def getRecall(self):
        return self.tp / (self.tp + self.fn)
    def getF1Score(self):
        return 2*self.tp / (2*self.tp + self.fp +self.fn)

    def getAverageDocPrecision(self):
        return self.average_doc_precision

    def getAverageDocRecall(self):
        return self.average_doc_recall

    def getAverageDocF1score(self):
        return self.average_doc_f1score

    def getAveragePrecision(self):
        return self.average_precision

    def getCombinedF1score(self):
        return self.combined_f1score

    def printResults(self):
        print("TP: " + str(self.getTp()))
        print("TN: " + str(self.getTn()))
        print("FP: " + str(self.getFp()))
        print("FN: " + str(self.getFn()))
        print("Accuracy: " + str(self.getAccuracy()))
        print("Precision: " + str(self.getPrecision()))
        print("Recall: " + str(self.getRecall()))
        print("F1Score: " + str(self.getF1Score()))
        print("Doc average precision: " + str(self.getDocAveragePrecision()))
        print("Doc average recall: " + str(self.getDocAverageRecall()))
        print("Doc average f1score: " + str(self.getDocAverageF1score()))
