import time
import os
import pandas as pd
import numpy as np

if __name__ == "__main__":
    # os.makedirs(os.path.dirname("output/"), exist_ok=True)
    start_time = time.time()

    dataframe = pd.read_csv('./datasets/train.tsv', sep='\t')
    test_dataframe = pd.read_csv('./datasets/test.tsv', sep='\t')
    A = np.array(dataframe)
    print(A)
    length = A.shape[0]
    print("Size of input: ", length)
    # print(dataframe)
    print("Total execution time %s seconds" % (time.time() - start_time))
