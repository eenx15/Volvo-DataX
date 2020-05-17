import ast
import pandas as pd
import numpy as np
import pickle
import tensorflow as tf
import math
import os
from sklearn.metrics import average_precision_score
from sklearn.model_selection import train_test_split

from keras.models import Sequential
from keras.layers import Bidirectional, BatchNormalization, LSTM, Dense, Dropout, TimeDistributed
from keras.preprocessing import sequence
from keras.layers import Masking

from pandas import DataFrame
from pandas import concat
from matplotlib import pyplot
from keras import losses


def get_length_data(data, max_frames):
    """
    Function that estimates the number of layers that the input data will be separated into
    :param data: Data file containing all input data
    :param max_frames: Nr of frames in each layer/sequence
    :return: An estimate of the number of layers.
    """
    nr_pedestrians = len(data['video_id'])
    nr_frames=[]

    for i in range(nr_pedestrians):
        if len(data['frame_number'][i]) == 0:
            continue
        curr_length_data = data['frame_number'][i][len(data['frame_number'][i]) - 1] - data['frame_number'][i][0]
        slack = curr_length_data % max_frames

        nr_whole_dims = (curr_length_data-slack)/max_frames
        if slack != 0:
            nr_tot_dims = nr_whole_dims + 1
        elif slack == 0 and nr_whole_dims != 0:
            nr_tot_dims = nr_whole_dims
        elif slack == 0 and nr_whole_dims == 0:
            nr_tot_dims = 1
        nr_frames.append(nr_tot_dims)
    return sum(nr_frames) + 2


def reformat_data (data, max_frames, data_type):
    """
    Reformats the input data into a feature data file and a target data file.
    :param data: Data file containing all input data
    :param max_frames: Nr of frames in each layer/sequence
    :param data_type: If it is train or test data
    :return:
        x_data - the feature data. Pedestrians in each layer, frame on each row (from 0), feature on each column
        y_data - the target data. Pedestrians in each layer, correct classification for each frame on each row (from 0).
    """
    relevant_columns = data.columns.tolist()

    for col_name in relevant_columns[2:]:
        data[col_name] = data[col_name].replace(np.nan, '[]')
        data[col_name] = data[col_name].astype(str)
        data[col_name] = data[col_name].apply(ast.literal_eval)

    attribute_list = relevant_columns[3:len(relevant_columns)-1]  # A list of all attributes we will recieve from the skeleton
    nr_dimensions = int(get_length_data(data, max_frames))  # Estimate the number of dimensions needed

    nr_pedestrians = len(data['video_id'])
    nr_inputs = 3 + len(attribute_list) # +3 since bounding box gives 4 input values

    y_data = np.zeros((nr_dimensions, max_frames, 1))
    x_data = np.full((nr_dimensions, max_frames, nr_inputs), -1.00000)

    # Add the box-coordinates to x_data
    add_on = 0
    for i in range(nr_pedestrians):
        index = 0

        # Skip empty sequences of data
        if len(data['frame_number'][i]) == 0:
            add_on -= 1
            continue

        j_start = data['frame_number'][i][0]
        j_old = 0

        for j in data['frame_number'][i]:
            j = j - j_start  # Shift the data so it is added into x_data from index 0

            # If we have filled one layer/sequence, continue adding the data to the next one
            if j >= max_frames:
                j_start = j + j_start
                add_on += 1
                j = 0

            box_vals = data['bounding_boxes'][i][index]  # The current bounding box values

            for k in range(4):
                x_data[i+add_on][j][k] = box_vals[k]/1500

            # Fills potential gaps in the data with the data from the first frame after the gap
            if j - j_old > 1:
                for diff in range(j - j_old):
                    x_data[i + add_on][j_old + diff][0:4] = x_data[i + add_on][j][0:4]
            j_old = j

            index += 1

    # Add the remaining skeleton-features to x_data.
    add_on = 0
    pie = math.pi
    for i in range(nr_pedestrians):
        k = 0
        add_on_curr = add_on

        # Iterate over all features in attribute_list except bounding_boxes
        for att in attribute_list[1:]:
            add_on = add_on_curr

            # Skip empty sequences of data
            if len(data['frame_number'][i]) == 0:
                add_on -= 1
                continue

            j_start = data['frame_number'][i][0]

            j_old = 0
            index = 0
            # Iterate over each data entry for that pedestrian and feature
            for j in data['frame_number'][i]:

                j = j - j_start  # Shift the data so it is added into x_data from index 0

                # If we have filled one layer/sequence, continue adding the data to the next one
                if j >= max_frames:
                    j_start = j + j_start
                    add_on += 1
                    j = 0

                att_val = data[att][i][index]

                if 'ang' in att:
                    data_add = att_val / pie  # Normalize
                    x_data[i + add_on][j][k + 4] = data_add
                elif 'Dist' in att:
                    x_data[i + add_on][j][k + 4] = att_val / 1500  # Normalize

                # Fills potential gaps in the data with the data from the first frame after the gap
                if j - j_old > 1:
                    for diff in range(j - j_old):
                        x_data[i + add_on][j_old + diff][k + 4] = x_data[i + add_on][j][k+4]
                j_old = j

                index += 1

            k += 1
    print('Skeleton-features done')

    # Add the target data to y_data
    add_on = 0
    for i in range(nr_pedestrians):
        index = 0

        # Skip empty sequences of data
        if len(data['frame_number'][i]) == 0:
            add_on -= 1
            continue

        j_start = data['frame_number'][i][0]
        j_old = 0

        for j in data['frame_number'][i]:

            j = j - j_start  # Shift the data so it is added into x_data from index 0

            # If we have filled one layer/sequence, continue adding the data to the next one
            if j >= max_frames:
                j_start = j + j_start
                add_on += 1
                j = 0

            y_data[i+add_on][j] = data['crossing'][i][index]

            # Fills potential gaps in the data with the data from the first frame after the gap
            if j - j_old > 1:
                for diff in range(j - j_old):
                    y_data[i + add_on][j_old + diff][0] = y_data[i + add_on][j]
            j_old = j

            index += 1
    print('Crossing done')

    pickle.dump(x_data[:i+add_on], open(f'data/x_data_2_{data_type}.pkl', 'wb'))
    pickle.dump(y_data[:i+add_on], open(f'data/y_data_2_{data_type}.pkl', 'wb'))

    return x_data[:i+add_on], y_data[:i+add_on]


def cross_in_future(Y_data, PT):
    """
    A function that adds P:s (positives) before each actual P in each sequence of the Target data, to enable earlier
    predictions.
    :param Y_data: Target data
    :param PT: Prediction Time. Number of P:s to add before each P.
    :return: The target data with extra P:s added
    """
    nr_ped = len(Y_data)
    nr_frame = len(Y_data[0])
    for i in range(nr_ped):
        for j in range(nr_frame):
            value = Y_data[i][j]
            if value == 1:
                for k in range(j-PT, j):
                    if k >= 0:
                        Y_data[i][j] = 1
    return Y_data


def evaluation_scores(Y_data, X_data, pred_data):
    """
    A function to evaluate the performance of the model by calculating different metrics. Excludes the padding and data
    from bounding boxes with a width smaller than 60 pixels. The number of P (positive) and N (negative) classification
    used to calculate the metrics are balanced.
    :param Y_data: Target data
    :param X_data: Feature data, used to identify padding
    :param pred_data: Prediction data, the binary predictions made by the model
    :return: The overall accuracy, the accuracy for P classifications, the accuracy for N classifications, the precision
    and the recall
    """

    nr_ped = len(Y_data)
    TP = 0
    FP = 0
    TN = 0
    FN = 0
    P = 0
    N = 0

    nr_P, nr_N = nr_P_N(pred_data, X_data)
    nr_PN = np.minimum(nr_P, nr_N)

    for i in range(nr_ped):
        for j in range(len(Y_data[i])):
            if X_data[i][j][0] != -1.0 and X_data[i][j][2]*1500 >= 60:
                if pred_data[i][j][0] == 0:
                    if N < nr_PN:
                        N += 1
                        if Y_data[i][j][0] == 0:
                            TN += 1
                        if Y_data[i][j][0] == 1:
                            FN += 1

    for i in range(nr_ped):
        for j in range(len(Y_data[i])):
            if X_data[i][j][0] != -1.0 and X_data[i][j][2]*1500 >= 60:
                if pred_data[i][j][0] == 1:
                    if P < nr_PN:
                        P += 1
                        if Y_data[i][j][0] == 1:
                            TP += 1
                        if Y_data[i][j][0] == 0:
                            FP += 1

    recall = TP/(TP+FN)
    precision = TP/(TP+FP)
    acc = (TP+TN)/(P+N)
    accP = TP / P
    accN = TN / N
    print("P: " + str(P))
    print("N: " + str(N))
    return acc, accP, accN, recall, precision


def nr_P_N(Y_data, X_data):
    """
    A function to calculate the total number of P:s (positives) and N:s (negatives) in the Target data. Excludes the
    padding and data from bounding boxes with a width smaller than 60 pixels.
    :param Y_data: Target data
    :param X_data: Feature data, used to identify padding
    :return: The total number of P:s and N:s
    """
    nr_ped = len(Y_data)
    nr_P = 0
    nr_N = 0
    for i in range(nr_ped):
        for j in range(len(Y_data[i])):
            if X_data[i][j][0] != -1.0 and X_data[i][j][2]*1500 >= 60:
                if Y_data[i][j][0] == 1:
                    nr_P += 1
                if Y_data[i][j][0] == 0:
                    nr_N += 1
    return nr_P, nr_N


def scheduler(epoch):
    """
    A function to decrease the learning rate over time.
    :param epoch: The current epoch
    :return: Learning rate
    """
    if epoch < 10:
        return float(0.001)
    else:
        return float(0.001 * tf.math.exp(0.1 * (10 - epoch)))


if __name__ == "__main__":

    print(os.path.basename(__file__))

    reload_data = True

    if reload_data:

        train_df = pd.read_csv('train_df_to_video_0250_trainedModel.csv')
        test_df = pd.read_csv('test_df_from_video_0251_trainedModel.csv')

        train_df = train_df.sort_values(by=['video_id'])
        test_df = test_df.sort_values(by=['video_id'])
        train_df = train_df.reset_index(drop=True)
        test_df = test_df.reset_index(drop=True)

        nr_frames = 45  # Nr of frames in each layer/sequence
        reformat_data(train_df, nr_frames, 'train')
        reformat_data(test_df, nr_frames, 'test')

    X_train = pickle.load(open('data/x_data_2_train.pkl', 'rb'))
    Y_train = pickle.load(open('data/y_data_2_train.pkl', 'rb'))
    X_test = pickle.load(open('data/x_data_2_test.pkl', 'rb'))
    Y_test = pickle.load(open('data/y_data_2_test.pkl', 'rb'))

    PT = 14

    # Add the PT to the sequences
    Y_train = cross_in_future(Y_train, PT)
    Y_test = cross_in_future(Y_test, PT)

    acc_all = []
    recall_all = []
    precision_all = []

    nr_runs = 1

    folder_nr = 1  # To name the folder to save all the data in

    # Find a folder name that is not occupied
    while os.path.exists(f'RNN_data_{folder_nr}'):
        folder_nr += 1

    os.makedirs(f'RNN_data_{folder_nr}')
    print("RNN Folder Number: " + str(folder_nr))

    for i in range(nr_runs):

        model = Sequential()

        EPOCHS = 50
        BATCH_SIZE = 8

        model.add(Bidirectional(LSTM(16, input_shape=(X_train[:, :, 4:].shape[1:]), return_sequences=True)))
        model.add(Dropout(0.1))
        model.add(BatchNormalization())

        model.add(LSTM(16, input_shape=(X_train[:, :, 4:].shape[1:]), return_sequences=True))
        model.add(BatchNormalization())

        model.add(LSTM(16, input_shape=(X_train[:, :, 4:].shape[1:]), return_sequences=True))
        model.add(BatchNormalization())

        model.add(TimeDistributed(Dense(1, activation='sigmoid')))
        model.add(Dropout(0.2))

        model.compile(loss='mae', optimizer='adam', metrics=['accuracy'])
        callback = tf.keras.callbacks.LearningRateScheduler(scheduler, verbose=0)

        # Training on the data in X_train, excluding the bounding boxes
        history = model.fit(X_train[:, :, 4:], Y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, callbacks=[callback],
                            validation_data=(X_test[:, :, 4:], Y_test), verbose=1)

        prediction_bin = model.predict_classes(X_test[:, :, 4:])
        prediction_prob = model.predict(X_test[:, :, 4:])

        (acc, accP, accN, recall, precision) = evaluation_scores(Y_test, X_test, prediction_bin)

        print("Run nr " + str(i+1))

        print("Accuracy: " + str(acc))
        print("Accuracy P: " + str(accP))
        print("Accuracy N: " + str(accN))
        print("Recall: " + str(recall))
        print("Precision: " + str(precision))

        acc_all.append(acc)
        recall_all.append(recall)
        precision_all.append(precision)

        # Save the best model
        if acc == np.amax(acc_all):
            model.save(f'RNN_data_{folder_nr}/model_best.h5')

        pyplot.figure(1)
        pyplot.plot(history.history['loss'], label='train loss')
        pyplot.plot(history.history['val_loss'], label='test val loss')
        title_str = ('Training and test data error with batch size equal to ' + str(BATCH_SIZE))
        pyplot.title(title_str)
        pyplot.legend()
        # pyplot.show()
        pyplot.savefig(f'RNN_data_{folder_nr}/test_train_plot_{i}.png')

        pyplot.figure(2)
        pyplot.plot(history.history['accuracy'], label='train accuracy')
        pyplot.plot(history.history['val_accuracy'], label='test val accuracy')
        title_str = ('Training and test data accuracy with batch size equal to ' + str(BATCH_SIZE))
        pyplot.title(title_str)
        pyplot.legend()
        # pyplot.show()
        pyplot.savefig(f'RNN_data_{folder_nr}/test_train_accuracy_plot_{i}.png')

    acc_average = np.sum(acc_all) / len(acc_all)
    recall_average = np.sum(recall_all) / len(recall_all)
    precision_average = np.sum(precision_all) / len(precision_all)

    print("Accuracy score NEW AVERAGE : " + str(acc_new_average))
    print("Recall AVERAGE: " + str(recall_average))
    print("Precision AVERAGE: " + str(precision_average))

