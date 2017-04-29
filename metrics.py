import math

class Metrics:
    def __init__(self):
        self.mi_accuracy = 0
        self.mi_f1score = 0
        self.mi_precision = 0
        self.mi_recall = 0
        self.ma_accuracy = 0
        self.ma_f1score = 0
        self.ma_precision = 0
        self.ma_recall = 0
        self.tp = 0
        self.tn = 0
        self.fp = 0
        self.fn = 0
        self.average_doc_precision = 0
        self.average_doc_recall = 0
        self.average_doc_f1score = 0
        self.combined_f1score = 0
        self.mean_average_precision = 0

        self.doc_precision_values = []
        self.doc_recall_values = []
        self.doc_f1score_values = []

    def updateMacroAverages(self, eval):
        self.ma_accuracy += eval.getAccuracy()
        self.ma_f1score += eval.getF1Score()
        self.ma_precision += eval.getPrecision()
        self.ma_recall += eval.getRecall()

        self.average_doc_precision += eval.getAverageDocPrecision()
        self.average_doc_recall += eval.getAverageDocRecall()
        self.average_doc_f1score += eval.getAverageDocF1score()
        self.combined_f1score += eval.getCombinedF1score()

        self.mean_average_precision += eval.getAveragePrecision()

        self.doc_precision_values.append(eval.getAverageDocPrecision())
        self.doc_recall_values.append(eval.getAverageDocRecall())
        self.doc_f1score_values.append(eval.getAverageDocF1score())

    def updateConfusionMatrix(self, eval):
        self.tp += eval.getTp()
        self.tn += eval.getTn()
        self.fp += eval.getFp()
        self.fn += eval.getFn()

    def calculate(self, test_set_size):
        self.ma_accuracy = self.ma_accuracy / test_set_size
        self.ma_f1score = self.ma_f1score / test_set_size
        self.ma_precision = self.ma_precision / test_set_size
        self.ma_recall = self.ma_recall / test_set_size
        self.mi_accuracy = (self.tp+self.tn)/(self.tp+self.tn+self.fp+self.fn)
        self.mi_precision = self.tp/(self.tp+self.fp)
        self.mi_recall = self.tp/(self.tp+self.fn)
        self.mi_f1score = 2*self.mi_precision*self.mi_recall/(self.mi_precision+self.mi_recall)

        self.average_doc_precision = self.average_doc_precision / test_set_size
        self.average_doc_recall = self.average_doc_recall / test_set_size
        self.average_doc_f1score = self.average_doc_f1score / test_set_size
        self.combined_f1score = self.combined_f1score / test_set_size

        self.mean_average_precision = self.mean_average_precision / test_set_size;

        prec_variance = 0
        rec_variance = 0
        f1_variance = 0
        for i in range(len(self.doc_precision_values)):
            prec_variance += self.doc_precision_values[i]*self.average_doc_precision
            rec_variance += self.doc_recall_values[i]*self.average_doc_recall
            f1_variance += self.doc_f1score_values[i]*self.average_doc_precision

        prec_variance /= test_set_size
        rec_variance /= test_set_size
        f1_variance /= test_set_size

        self.doc_precision_std_dev = math.sqrt(prec_variance)
        self.doc_recall_std_dev = math.sqrt(rec_variance)
        self.doc_f1score_std_dev = math.sqrt(f1_variance)
