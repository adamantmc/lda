import os;

class FileWriter:
    def __init__(self):
        self.filenames = ["results/ma_accuracy.txt",
                          "results/ma_f1score.txt",
                          "results/ma_precision.txt",
                          "results/ma_recall.txt",
                          "results/mi_accuracy.txt",
                          "results/mi_f1score.txt",
                          "results/mi_precision.txt",
                          "results/mi_recall.txt"]

    if not os.path.exists("results"):
        os.makedirs("results")
    if not os.path.exists("results/queries"):
        os.makedirs("results/queries")

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

        for file in files:
            file.close()

    def writeQueryResults(self, results, query_id):
        file = open("results/queries/"+str(query_id)+".txt", 'w')
        for result in results:
            file.write(result[1]["title"]+"\n")
        file.close()