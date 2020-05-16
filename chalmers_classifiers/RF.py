# The RF-classifer code is based on pedestrian_crossing_predictions by Wade Rosko, https://github.com/wrosko/pedestrian_crossing_predictions

import pandas as pd
import numpy as np
import ast
from sklearn.model_selection import train_test_split
from sklearn.ensemble.forest import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import accuracy_score
from joblib import dump, load
import argparse
import random



def search_parameters(features, target, classifier):
    """
    To search for the optimal parameters for the RF-classifier.
    Parameters
    ----------
    features - X-data
    target - Y-data
    classifier - RandomForestClassifier object

    Returns
    -------
    GridSearchCV object
    """
    param_grid = {
        'n_estimators': [400, 500, 600, 700],
        'max_depth': [15, 40, 50, 60]
    }

    gdcv = GridSearchCV(estimator=classifier, param_grid=param_grid, cv=5)

    gdcv.fit(features, target)
    return gdcv


def random_forest_classifier(features, target):
    """
    To train the random forest classifier with features and target data
    :param features:
    :param target:
    :return: trained random forest classifier
    """
    clf = RandomForestClassifier(n_estimators=600, max_depth=50)
    clf.fit(features, target)
    return clf


def GenerateData(moving_window_size, pedestrians_df, isTrain):
    """
    Generates data formatted for RF
    :param moving_window_size: Nr of frames to base prediction on
    :param pedestrians_df: dataframe loaded from csv-file, containing features from skeletons
    :param isTrain: if True, balance the data (same number of C as NC)
    :return: tuple of xdata,ydata
    """
    count = 0
    data = []

    '''
    In order to use the data in the pedestrians_df dataframe, we use ast.literal_eval to evaluate the strings
    '''
    fieldnames = ['frame_number']
    for i in range(252):
        fieldnames.append('ang3-' + str(i))
    for i in range(36):
        fieldnames.append('ang-' + str(i))
    for i in range(36):
        fieldnames.append('xDist-' + str(i))
    for i in range(36):
        fieldnames.append('yDist-' + str(i))
    for i in range(36):
        fieldnames.append('sDist-' + str(i))
    fieldnames.append('bounding_boxes')
    fieldnames.append('crossing')

    for col_name in fieldnames:
        pedestrians_df[col_name] = pedestrians_df[col_name].replace(np.nan, '0')
        pedestrians_df[col_name] = pedestrians_df[col_name].astype(str)
        pedestrians_df[col_name] = pedestrians_df[col_name].apply(ast.literal_eval)

    pedestrian_data = []
    result_data = []

    balanced_pedestrian_data = []
    balanced_result_data = []

    '''
    attribute_list will contain the names of the 396 features, for example ang3-0 which corresponds
    to the first ang3 feature in the csv-file for each pedestrian in each frame
    '''
    attribute_list = []
    for i in range(252):
        attribute_list.append('ang3-' + str(i))
    for i in range(36):
        attribute_list.append('ang-' + str(i))
    for i in range(36):
        attribute_list.append('xDist-' + str(i))
    for i in range(36):
        attribute_list.append('yDist-' + str(i))
    for i in range(36):
        attribute_list.append('sDist-' + str(i))


    '''
    Loop through the pedestrians in the dataframe
    '''
    for ped_n in range(len(pedestrians_df)):
        pedestrian = pedestrians_df.iloc[ped_n]
        n_attributes = len(attribute_list)
        n_frames = len(pedestrian['frame_number'])

        # Generate empty list of lists to store all data for pedestrian in
        general_attribute_list = [[] for _ in range(n_attributes)]

        # Insert the classified attributes from the dataframe into the general attribute list
        for j in range(n_attributes):
            attribute = attribute_list[j]

            empty_array = np.zeros(n_frames)

            if isinstance(pedestrians_df[attribute][ped_n], list):
                counter = len(pedestrians_df[attribute][ped_n])
            else:
                counter = pedestrians_df[attribute][ped_n]

            for i in range(counter):
                empty_array[int(i)] = pedestrians_df[attribute][ped_n][int(i)]

            general_attribute_list[j] = empty_array

        general_attribute_list = np.array(general_attribute_list)

        general_attribute_list = np.multiply(general_attribute_list, 1)

        pedestrian = pedestrians_df.iloc[ped_n]

        '''
        Excluding pedestrians whose bounding box is less than 60px wide
        '''
        for frame in range(len(pedestrian['frame_number'])):
            if pedestrian['bounding_boxes'][frame][2] >= 60:
                features = createFeatures(moving_window_size, general_attribute_list, frame)
                if isinstance(features, int):
                    if features == -1:
                        continue
                pedestrian_data.append(features)
                result_data.append(pedestrian['crossing'][frame])
                count += 1

    crossIdx = []
    noCrossIdx = []

    '''
    Here the balancing takes place
    '''
    if isTrain:
        for cross in range(len(result_data)):
            if result_data[cross] == 1:
                crossIdx.append(cross)
            elif result_data[cross] == 0:
                noCrossIdx.append(cross)
        if len(crossIdx) > len(noCrossIdx):
            selected_cross_idx = random.sample(crossIdx, len(noCrossIdx))
            for idx in selected_cross_idx:
                balanced_pedestrian_data.append(pedestrian_data[idx])
                balanced_result_data.append(result_data[idx])
            for idx in noCrossIdx:
                balanced_pedestrian_data.append(pedestrian_data[idx])
                balanced_result_data.append(result_data[idx])

        if len(crossIdx) < len(noCrossIdx):
            selected_no_cross_idx = random.sample(noCrossIdx, len(crossIdx))
            for idx in selected_no_cross_idx:
                balanced_pedestrian_data.append(pedestrian_data[idx])
                balanced_result_data.append(result_data[idx])
            for idx in crossIdx:
                balanced_pedestrian_data.append(pedestrian_data[idx])
                balanced_result_data.append(result_data[idx])


        print(count)
        print(len(crossIdx))
        print(len(noCrossIdx))
        print(len(balanced_pedestrian_data))
        print(len(balanced_result_data))
        out_data = np.asarray(np.array(balanced_pedestrian_data))
        out_result_data = np.asarray(np.matrix(balanced_result_data).T)

    else:
        out_data = np.asarray(np.array(pedestrian_data))
        out_result_data = np.asarray(np.matrix(result_data).T)
    return (out_data, out_result_data)


def createFeatures(moving_window_size, general_attribute_list, frame):
    """
    This function extracts features from the general_attribute_list given the moving_window_size and the current frame
    :param moving_window_size: Nr of frames to base prediction on
    :param general_attribute_list: Contains the features for a specific pedestrian, for all frames that the pedestrian
    is visible
    :param frame:current frame to create features for
    :returns Array of features within the moving window size (396*T features, where T is moving window size)
    """
    result = []
    padded = False
    for movingFrame in range(moving_window_size):
        if frame - movingFrame < 0:
            padded = True
            result.append(general_attribute_list[:, 0])
        else:
            result.append(general_attribute_list[:, frame - movingFrame])

    if padded:
        return -1
    else:
        return np.array(result).ravel()


def parse_args():
    """ Parse command line arguments.
    """
    parser = argparse.ArgumentParser(description="RF")

    parser.add_argument(
        "--grid_search", help="search for best parameters",
        default=0, required=False)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    ''' Uncomment if run with a single csv file
    pedestrians_df = pd.read_csv('feature_file_next_frame_v2.csv')
    (Xdata, Ydata) = GenerateData(14, pedestrians_df)
    

    X_train, X_test, y_train, y_test = train_test_split(Xdata, Ydata, test_size=0.3)
    '''

    # Run with two csv-files, splitted in train and test
    pedestrians_df_train = pd.read_csv('train_features_0.7.csv')
    (X_train, y_train) = GenerateData(14, pedestrians_df_train, isTrain=True)

    pedestrians_df_test = pd.read_csv('test_features_0.7.csv')
    (X_test, y_test) = GenerateData(14, pedestrians_df_test, isTrain=False)

    y_train = np.ravel(y_train)
    y_test = np.ravel(y_test)


    train_accuracy = []
    test_accuracy = []

    if args.grid_search:
        grid = search_parameters(X_train, y_train, RandomForestClassifier())
        predictions = grid.best_estimator_.predict(X_test)
        print(accuracy_score(y_test, predictions))
        dump(grid.best_estimator_, 'grid_searched_27_4.pkl')
    else:
        # Run for X times
        for i in range(1):
            print("Running iteration " + str(i))
            trained_model = random_forest_classifier(X_train, y_train)

            predictions = trained_model.predict(X_test)
            train_accuracy.append(accuracy_score(y_train, trained_model.predict(X_train)))
            test_accuracy.append(accuracy_score(y_test, predictions))

        print("Trained model :: ", trained_model)
        predict = trained_model.predict(X_test)
        print(accuracy_score(y_test, predict))
        dump(trained_model, 'trainedRF_common_split.joblib')  # Save trained model on disk


