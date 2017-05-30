import time
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import savefig
import matplotlib.mlab as mlab



def createPlots(dataframe):
    attributeType = ["qualitative", "numerical", "qualitative", "qualitative", "numerical", "qualitative", "qualitative",
            "numerical", "qualitative", "qualitative", "numerical", "qualitative", "numerical", "qualitative",
            "qualitative", "numerical", "qualitative", "numerical", "qualitative", "qualitative"]
    good = dataframe[dataframe["Label"] == 1]
    bad = dataframe[dataframe["Label"] == 2]
    i = 0
    for column in dataframe:
        if i == 20:
            break
        if attributeType[i] == "qualitative":
            # plt.plot(good['column'].value_counts()) den doulevei
            good[column].value_counts().plot(kind='bar')
            # plt.xlabel('Age')
            # plt.ylabel('# of Applicants')
            name = "output/Attribute" + str(i + 1) + "_" + "good.png"
            savefig(name)
            plt.figure()
            bad[column].value_counts().plot(kind='bar')
            name = "output/Attribute" + str(i + 1) + "_" + "bad.png"
            savefig(name)
            # plt.xlabel('Age')
            # plt.ylabel('# of Applicants')
            if i < 19:
                plt.figure()
        elif attributeType[i] == "numerical":
            good.boxplot(column)
            # plt.xlabel('Age')
            # plt.ylabel('# of Applicants')
            name = "output/Attribute" + str(i + 1) + "_" + "good.png"
            savefig(name)
            plt.figure()
            bad.boxplot(column)
            # plt.xlabel('Age')
            # plt.ylabel('# of Applicants')
            name = "output/Attribute" + str(i + 1) + "_" + "bad.png"
            savefig(name)
            if i < 19:
                plt.figure()
        i += 1
    # plt.show()


if __name__ == "__main__":
    os.makedirs(os.path.dirname("output/"), exist_ok=True)
    start_time = time.time()
    dataframe = pd.read_csv('./datasets/train.tsv', sep='\t')
    test_dataframe = pd.read_csv('./datasets/test.tsv', sep='\t')
    A = np.array(dataframe)
    print(A)
    length = A.shape[0]
    print("Size of input: ", length)
    # print(dataframe)
    # print(dataframe["Attribute1"])
    createPlots(dataframe)
    print("Total execution time %s seconds" % (time.time() - start_time))
