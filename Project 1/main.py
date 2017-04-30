%matplotlib inline

import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np



# the histogram of the data
plt.hist(df["Age (years)"], facecolor='green')
plt.xlabel('Age')
plt.ylabel('# of Applicants')
plt.show()

if __name__ == "__main__":
    df = pd.read_csv('./Documentation/train_set.csv', sep='\t')
    # for column in df.columns:
    #     print(column)
    # print(df.describe())
    # print(df[["RowNum", "Content", "Category"]].iloc[5:10])
    # print(df[df["RowNum"] < 5])

    A = np.array(df)
    print(A.shape)

    for i in range(A.shape[0]):
        text = ""
        for j in range(A.shape[1]):
            text += str(A[i, j]) + ","
        print(text)
