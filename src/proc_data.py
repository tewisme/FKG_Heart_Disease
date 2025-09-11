import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def visualization(data):
    dataF = data.copy()
    featureName = dataF.columns.tolist()
    for i in range(1):
        plt.figure(figsize=(11,4))
        plt.subplot(1,3,1)
        sns.histplot(dataF[featureName[i]], bins=10, kde=True)
        plt.title('Histplot diagram for ' + str(featureName[i]))
        #---
        plt.subplot(1,3,2)
        sns.boxplot(x='target', y=featureName[i], data=dataF)
        sns.stripplot(x='target', y=featureName[i], data=dataF, jitter=True, edgecolor='black')
        plt.title('Boxplot for ' + str(featureName[i]))
        #---
        plt.subplot(1,3,3)
        sns.violinplot(x='target', y=featureName[i], data=dataF)
        sns.stripplot(x='target', y=featureName[i], data=dataF, jitter=True, edgecolor='black')
        plt.title('Violinplot for ' + str(featureName[i]))
    plt.show()
