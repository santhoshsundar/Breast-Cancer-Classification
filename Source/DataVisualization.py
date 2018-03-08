import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# Normalize dataset
def normalize(modified_breast_cancer_data):
    normalized_breast_cancer_data = modified_breast_cancer_data.copy()
    for feature_name in modified_breast_cancer_data.columns:
        max_value = modified_breast_cancer_data[feature_name].max()
        min_value = modified_breast_cancer_data[feature_name].min()
        normalized_breast_cancer_data[feature_name] = (modified_breast_cancer_data[feature_name] - min_value) / (max_value - min_value)
    return normalized_breast_cancer_data

# Number of Benign and Malignant Diagnosis
def category_count(labels):
    B, M = labels.value_counts()
    print('Number of Benign: ',B)
    print('Number of Malignant : ',M)
    sns.countplot(labels)
    plt.title('Number of Benign and Malignant Diagnosis')
    plt.xlabel('Diagnosis')
    plt.ylabel('Count')
    plt.show()

# Draw Violin Plot
def violin_plot(normalized_breast_cancer_data,labels):

    fig = plt.figure(figsize=(15,8))
    gs = GridSpec(3, 10)
    ax1 = fig.add_subplot(gs[0,:])
    ax2 = fig.add_subplot(gs[1,:])
    ax3 = fig.add_subplot(gs[2,:])

    label_1 = normalized_breast_cancer_data.iloc[:, 0:10].columns.values
    label_2 = normalized_breast_cancer_data.iloc[:, 10:20].columns.values
    label_3 = normalized_breast_cancer_data.iloc[:, 20:31].columns.values

    data_1 = pd.concat([labels, normalized_breast_cancer_data.iloc[:, 0:10]], axis=1)
    melted_data_1 = pd.melt(data_1, id_vars="diagnosis",
                            var_name="features",
                            value_name='value')
    sns.violinplot(x="features", y="value", hue="diagnosis", data=melted_data_1, split=True, inner="quart", ax=ax1)
    labels_1 = ax1.set_xticklabels(label_1)
    for i, label in enumerate(labels_1):
        label.set_y(label.get_position()[1] - (i % 2) * 0.075)
    ax1.plot()

    data_2 = pd.concat([labels, normalized_breast_cancer_data.iloc[:, 10:20]], axis=1)
    melted_data_2 = pd.melt(data_2, id_vars="diagnosis",
                            var_name="features",
                            value_name='value')
    sns.violinplot(x="features", y="value", hue="diagnosis", data=melted_data_2, split=True, inner="quart",ax=ax2)
    labels_2 = ax2.set_xticklabels(label_2)
    for i, label in enumerate(labels_2):
        label.set_y(label.get_position()[1] - (i % 2) * 0.075)
    ax2.plot()

    data_3 = pd.concat([labels, normalized_breast_cancer_data.iloc[:, 20:31]], axis=1)
    melted_data_3 = pd.melt(data_3, id_vars="diagnosis",
                            var_name="features",
                            value_name='value')
    sns.violinplot(x="features", y="value", hue="diagnosis", data=melted_data_3, split=True, inner="quart",ax=ax3)
    labels_3 = ax3.set_xticklabels(label_3)
    for i, label in enumerate(labels_3):
        label.set_y(label.get_position()[1] - (i % 2) * 0.075)
    ax3.plot()

    fig.subplots_adjust(top=.98)

    plt.show()

# Draw Swarm Plot
def swarm_plot(normalized_breast_cancer_data,labels):
    fig = plt.figure(figsize=(15, 8))
    gs = GridSpec(3, 10)
    ax1 = fig.add_subplot(gs[0, :])
    ax2 = fig.add_subplot(gs[1, :])
    ax3 = fig.add_subplot(gs[2, :])

    label_1 = normalized_breast_cancer_data.iloc[:, 0:10].columns.values
    label_2 = normalized_breast_cancer_data.iloc[:, 10:20].columns.values
    label_3 = normalized_breast_cancer_data.iloc[:, 20:31].columns.values

    data_1 = pd.concat([labels, normalized_breast_cancer_data.iloc[:, 0:10]], axis=1)
    melted_data_1 = pd.melt(data_1, id_vars="diagnosis",
                            var_name="features",
                            value_name='value')
    sns.swarmplot(x="features", y="value", hue="diagnosis", data=melted_data_1,dodge=True, ax=ax1)
    labels_1 = ax1.set_xticklabels(label_1)
    for i, label in enumerate(labels_1):
        label.set_y(label.get_position()[1] - (i % 2) * 0.075)
    ax1.plot()

    data_2 = pd.concat([labels, normalized_breast_cancer_data.iloc[:, 10:20]], axis=1)
    melted_data_2 = pd.melt(data_2, id_vars="diagnosis",
                            var_name="features",
                            value_name='value')
    sns.swarmplot(x="features", y="value", hue="diagnosis", data=melted_data_2,dodge=True, ax=ax2)
    labels_2 = ax2.set_xticklabels(label_2)
    for i, label in enumerate(labels_2):
        label.set_y(label.get_position()[1] - (i % 2) * 0.075)
    ax2.plot()

    data_3 = pd.concat([labels, normalized_breast_cancer_data.iloc[:, 20:31]], axis=1)
    melted_data_3 = pd.melt(data_3, id_vars="diagnosis",
                            var_name="features",
                            value_name='value')
    sns.swarmplot(x="features", y="value", hue="diagnosis", data=melted_data_3,dodge=True, ax=ax3)
    labels_3 = ax3.set_xticklabels(label_3)
    for i, label in enumerate(labels_3):
        label.set_y(label.get_position()[1] - (i % 2) * 0.075)
    ax3.plot()

    fig.subplots_adjust(top=.98)

    plt.show()


if __name__ == "__main__":

    breast_cancer_data = pd.read_csv('../BreastCancer.csv', delimiter=',')
    labels = breast_cancer_data.diagnosis
    list = ['Unnamed: 32', 'id', 'diagnosis']
    modified_breast_cancer_data = breast_cancer_data.drop(list, axis=1)

    category_count(labels)

    normalized_breast_cancer_data  = normalize(modified_breast_cancer_data)

    violin_plot(normalized_breast_cancer_data,labels)

    swarm_plot(normalized_breast_cancer_data,labels)