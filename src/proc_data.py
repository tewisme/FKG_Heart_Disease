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

def relationship(data):
    dataF = data.copy()
    dataF.hist(figsize=(12,8), bins=20)
    plt.suptitle("histogram of Features")
    plt.show()
    featureName = ["chest pain type", "exercise angina", "oldpeak", "ST slope", "max heart rate"]
    for i in range(len(featureName)-1):
        for j in range(i+1, len(featureName)-2):
            sns.jointplot(x=featureName[i], y=featureName[j], data = dataF, size = 7, hue="target")
            plt.title("Relationship between " + str(featureName[i]) + ' ' + str(featureName[j]))
            #---
            plt.show()
def heatmap(data):
    dataF = data.copy()
    plt.figure(figsize=(12,8))
    sns.heatmap(dataF.corr(), annot=False, cmap="coolwarm")
    plt.title("Feature Correlation Heatmap")
    plt.show()
def call(data):
    #heatmap(data)
    relationship(data)
