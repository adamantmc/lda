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

    def updateMacroAverages(self, u_accuracy, u_f1score, u_precision, u_recall):
        self.ma_accuracy += u_accuracy
        self.ma_f1score += u_f1score
        self.ma_precision += u_precision
        self.ma_recall += u_recall

    def updateConfusionMatrix(self, tp, tn, fp, fn):
        self.tp += tp
        self.tn += tn
        self.fp += fp
        self.fn += fn

    def calculate(self, test_set_size):
        self.ma_accuracy = self.ma_accuracy / test_set_size
        self.ma_f1score = self.ma_f1score / test_set_size
        self.ma_precision = self.ma_precision / test_set_size
        self.ma_recall = self.ma_recall / test_set_size
        self.mi_accuracy = (self.tp+self.tn)/(self.tp+self.tn+self.fp+self.fn)
        self.mi_precision = self.tp/(self.tp+self.fp)
        self.mi_recall = self.tp/(self.tp+self.fn)
        self.mi_f1score = 2*self.mi_precision*self.mi_recall/(self.mi_precision+self.mi_recall)

