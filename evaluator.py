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

        retrieved_labels = set()

        for doc in self.retrieved_docs:
            for label in doc["meshMajor"]:
                retrieved_labels.add(label)

        for label in self.query_doc["meshMajor"]:
            if label in retrieved_labels:
                self.tp = self.tp + 1
                retrieved_labels.remove(label)
            else:
                self.fn = self.fn + 1

        self.fp = len(retrieved_labels)
        self.tn = self.all_label_count - self.fp - self.fn - self.tp

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

    def printResults(self):
        print("TP: " + str(eval.getTp()))
        print("TN: " + str(eval.getTn()))
        print("FP: " + str(eval.getFp()))
        print("FN: " + str(eval.getFn()))
        print("Accuracy: " + str(eval.getAccuracy()))
        print("Precision: " + str(eval.getPrecision()))
        print("Recall: " + str(eval.getRecall()))
        print("F1Score: " + str(eval.getF1Score()))