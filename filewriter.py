import os

class FileWriter:
    def __init__(self, dir = "results"):
        self.dir = dir
        if not os.path.exists(dir):
            os.makedirs(dir)

        self.filenames = [self.dir+"/ma_accuracy.txt",
                          self.dir+"/ma_f1score.txt",
                          self.dir+"/ma_precision.txt",
                          self.dir+"/ma_recall.txt",
                          self.dir+"/mi_accuracy.txt",
                          self.dir+"/mi_f1score.txt",
                          self.dir+"/mi_precision.txt",
                          self.dir+"/mi_recall.txt",
                          self.dir+"/doc_average_precision.txt",
                          self.dir+"/doc_average_recall.txt",
                          self.dir+"/doc_average_f1score.txt",
                          self.dir+"/combined_f1score.txt",
                          self.dir+"/mean_average_precision.txt"]

    def writeToFiles(self, metrics_obj_list, thresholds):
        files = []
        for file in self.filenames:
            files.append(open(file, 'w'))

        for i in range(0, len(thresholds)):
            files[0].write(str(thresholds[i]) + " " + str(metrics_obj_list[i].ma_accuracy) + "\n")
            files[1].write(str(thresholds[i]) + " " + str(metrics_obj_list[i].ma_f1score) + "\n")
            files[2].write(str(thresholds[i]) + " " + str(metrics_obj_list[i].ma_precision) + "\n")
            files[3].write(str(thresholds[i]) + " " + str(metrics_obj_list[i].ma_recall) + "\n")
            files[4].write(str(thresholds[i]) + " " + str(metrics_obj_list[i].mi_accuracy) + "\n")
            files[5].write(str(thresholds[i]) + " " + str(metrics_obj_list[i].mi_f1score) + "\n")
            files[6].write(str(thresholds[i]) + " " + str(metrics_obj_list[i].mi_precision) + "\n")
            files[7].write(str(thresholds[i]) + " " + str(metrics_obj_list[i].mi_recall) + "\n")
            files[8].write(str(thresholds[i]) + " " + str(metrics_obj_list[i].average_doc_precision) + " " + str(metrics_obj_list[i].doc_precision_std_dev) + "\n")
            files[9].write(str(thresholds[i]) + " " + str(metrics_obj_list[i].average_doc_recall) + " " + str(metrics_obj_list[i].doc_recall_std_dev)    + "\n")
            files[10].write(str(thresholds[i]) + " " + str(metrics_obj_list[i].average_doc_f1score) + " " + str(metrics_obj_list[i].doc_f1score_std_dev) + "\n")
            files[11].write(str(thresholds[i]) + " " + str(metrics_obj_list[i].combined_f1score) + "\n")
            files[12].write(str(thresholds[i]) + " " + str(metrics_obj_list[i].mean_average_precision) + "\n")

        for file in files:
            file.close()

    def writeQueryResults(self, results, query_id):
        file = open(self.dir+"/queries/"+str(query_id)+".txt", 'w')
        for result in results:
            file.write(result[1]["title"]+"\n")
        file.close()
