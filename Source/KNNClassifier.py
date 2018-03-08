import pandas as pd
import numpy as np
import random
from sklearn.model_selection import train_test_split

K_CROSS = 10
nearest_neighbours = [i for i in range(5, 12, 2)]

# Normalize dataset
def normalize(breast_cancer_data):
    normalized_data = breast_cancer_data.copy()
    for feature_name in breast_cancer_data.columns:
        max_value = breast_cancer_data[feature_name].max()
        min_value = breast_cancer_data[feature_name].min()
        normalized_data[feature_name] = (breast_cancer_data[feature_name] - min_value) / (max_value - min_value)
    return normalized_data

# Split train dataset into training and validate datasets
def train_validate_split(x_train,y_train,K_CROSS):
    trains_x = []
    trains_y = []
    k_cross_size = len(x_train) * (K_CROSS/100)

    for k_cross in range(K_CROSS):
        train_x = []
        train_y = []
        while len(train_x) < k_cross_size:
            index = random.randrange(len(x_train))
            train_x.append(x_train.iloc[index])
            train_y.append(y_train.iloc[index])
        trains_x.append(pd.DataFrame(train_x))
        trains_y.append(pd.DataFrame(train_y))

    return trains_x, trains_y

# Find Majority of class labels
def majority(labels):
    count_0 = 0
    count_1 = 0

    for i in range(len(labels)):
        if (labels[i] == 1):
            count_1 += 1
        else:
            count_0 += 1

    if (count_0 > count_1):
        return 0
    else:
        return 1

# Determine label of a validate set
def knn_predict(x_train, y_train, x_validate, neighbour):
    distances = []
    labels = []

    for index in range(len(x_train)):

        # Calculating distances between train and test data using Eculidian Distance
        distance = np.sqrt(np.sum(np.square(np.array(x_validate),np.array( x_train.iloc[index,:]))))
        distances.append([distance,index])

    sorted_distances = sorted(distances)

    for i in range(neighbour):
        label_index = sorted_distances[i][1]
        labels.append(y_train.iloc[label_index][0])

    return majority(labels)

# Choose optimal nearest neighbour
def knn_nearest_neighbours_select(x_train,y_train):

    train_validate_data, train_validate_label = train_validate_split(x_train, y_train, K_CROSS)

    df_train = pd.DataFrame()
    df_label = pd.DataFrame()

    train_x = []
    train_y = []
    predictions = {}
    fold_prediction = {}

    for nearest_neighbour in range(len(nearest_neighbours)) :

        neighbour = nearest_neighbours[nearest_neighbour]
        print("Start of neighbour {}".format(neighbour))
        fold_predictions = {}
        fold_cross = 1

        for i in range(2):
            print("Start of validation {}".format(i))
            prediction = []

            x_validate = train_validate_data[i]
            y_validate = train_validate_label[i]

            train_x.clear()
            train_y.clear()

            for j in range(0,i):
                train_x.append(train_validate_data[j])
                train_y.append(train_validate_label[j])

            for k in range(i+1,len(train_validate_data)):
                train_x.append(train_validate_data[k])
                train_y.append(train_validate_label[k])

            df_train = pd.concat(train_x)
            df_label = pd.concat(train_y)

            for length in range(len(x_validate)):
                prediction.append([knn_predict(df_train,df_label,x_validate.iloc[length,:],neighbour),y_validate.iloc[length,:][0]])

            fold_predictions[fold_cross] = prediction
            print(fold_predictions)
            fold_cross += 1

        fold_list = []
        for key in fold_predictions.keys():
            accuracy = 0
            value_lists = fold_predictions[key]
            for list in value_lists:
                if list[0] == list[1]:
                    accuracy += 1
            fold_list.extend(accuracy/len(value_lists))

        fold_prediction[neighbour] = np.mean(fold_list)

    sorted_predictions = sorted(fold_prediction,key= lambda x: fold_prediction[x])

    for key in predictions.keys():
        print("Accuracy of Classifier with {0} Nearest Neighbours is : {1} ".format([key,sorted_predictions[key]]))

    optimal_nearest_neighbour = sorted_predictions.keys()[0]

    return optimal_nearest_neighbour

# Determine label of test dataset with optimal nearest neighbour
def knn(x_train,y_train,x_test,y_test,k_nearest_neighbour):
    predictions = {}
    prediction = []

    for length in range(len(x_test)):
        prediction.append([knn_predict(x_train,y_train,x_test.iloc[length,:],k_nearest_neighbour),y_test.iloc[length,:]])

    predictions[k_nearest_neighbour] = prediction

    value_lists = predictions[k_nearest_neighbour]
    for list in value_lists:
        accuracy = 0
        if list[0] == list[1]:
            accuracy += 1

    accuracy=(accuracy/len(value_lists))

    print("Accuracy Percentage of Classifier with {0} Nearest Neighbours is : {1} ".format([k_nearest_neighbour,accuracy]))

if __name__ == "__main__":

    breast_cancer_data = pd.read_csv('../BreastCancer.csv', delimiter=',')
    breast_cancer_data['diagnosis'] = breast_cancer_data['diagnosis'].map({'M': 1, 'B': 0})
    breast_cancer_data = breast_cancer_data.iloc[:, 1:32]
    data_x = breast_cancer_data.iloc[:, 2:]
    normalized_data_x = normalize(data_x)
    label_y = breast_cancer_data.iloc[:, 0]

    x_train, x_test, y_train, y_test = train_test_split(normalized_data_x, label_y, test_size=0.12126, random_state=42)

    optimal_nearest_neighbours = knn_nearest_neighbours_select(x_train, y_train)

    knn(x_train, y_train, x_test, y_test,optimal_nearest_neighbours)