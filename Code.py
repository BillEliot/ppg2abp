import csv
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import sklearn.metrics as skm
import math

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Conv1D, MaxPool1D, Dropout, Flatten
from tensorflow.keras.layers import RepeatVector, TimeDistributed

def split_dataset(data_PPG, data_ABP):
    split = int(len(data_PPG) * 0.8)
    train_PPG, test_PPG = np.array(data_PPG[ : split]), np.array(data_PPG[split :])
    train_ABP, test_ABP = np.array(data_ABP[ : split]), np.array(data_ABP[split :])
    
    return train_PPG, test_PPG, train_ABP, test_ABP

def sliding_window(train_PPG, train_ABP, sw_width):
    x, y = [], []
    
    in_start = 0
    for _ in range(len(train_PPG) - sw_width):
        in_end = in_start + sw_width

        train_seq = train_PPG[in_start : in_end]
        train_seq = train_seq.reshape((len(train_seq), 1))
        x.append(train_seq)
        y.append(train_ABP[in_start : in_end])
        
        in_start += 1
        
    return np.array(x), np.array(y)

def ConstructModel():
    model = Sequential()
    model.add(Conv1D(filters=64, kernel_size=3, padding='SAME', activation='relu', input_shape=(sliding_window_width, 1)))
    model.add(Conv1D(filters=96, kernel_size=3, padding='SAME', activation='relu'))
    model.add(MaxPool1D(pool_size=2))
    model.add(Flatten())
    model.add(RepeatVector(sliding_window_width))
    model.add(LSTM(100, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(200, return_sequences=True))
    model.add(TimeDistributed(Dense(90, activation='relu')))
    model.add(TimeDistributed(Dense(1)))
    
    return model

if __name__ == '__main__':
    data_PPG = []
    data_ABP = []

    with open('/home/Data/Dataset.csv', 'r') as f:
        csv_data = list(csv.reader(f))
        for value in csv_data[0][:]:
            data_PPG.append(float(value))
        for value in csv_data[1][:]:
            data_ABP.append(float(value))

    train_PPG, test_PPG, train_ABP, test_ABP = split_dataset(data_PPG, data_ABP)
    scaler_PPG = MinMaxScaler().fit([[min(data_PPG)], [max(data_PPG)]])
    scaler_ABP = MinMaxScaler().fit([[min(data_ABP)], [max(data_ABP)]])

    for sliding_window_width in [62, 125, 187, 250, 312, 375]:
        rmse_last = 99999
        for fold in range(10):
            fold_train_PPG = train_PPG[int(len(train_PPG) * fold * 0.1) : int(len(train_PPG) * (fold + 1) * 0.1)]
            fold_train_ABP = train_ABP[int(len(train_ABP) * fold * 0.1) : int(len(train_ABP) * (fold + 1) * 0.1)]
            fold_test_PPG = np.concatenate((train_PPG[ : int(len(train_PPG) * fold * 0.1)], train_PPG[int(len(train_PPG) * (fold + 1) * 0.1) : ]))
            fold_test_ABP = np.concatenate((train_ABP[ : int(len(train_ABP) * fold * 0.1)], train_ABP[int(len(train_ABP) * (fold + 1) * 0.1) : ]))

            fold_train_PPG = scaler_PPG.transform(fold_train_PPG.reshape(-1, 1))
            fold_train_ABP = scaler_ABP.transform(fold_train_ABP.reshape(-1, 1))
            fold_test_PPG = scaler_PPG.transform(fold_test_PPG.reshape(-1, 1))
            fold_test_ABP = scaler_ABP.transform(fold_test_ABP.reshape(-1, 1))

            fold_train_x, fold_train_y = sliding_window(fold_train_PPG, fold_train_ABP, sliding_window_width)
            fold_train_y = fold_train_y.reshape((fold_train_y.shape[0], fold_train_y.shape[1], 1))


            model = ConstructModel()
            model.compile(loss='mse', optimizer='adam')
            model.fit(fold_train_x, fold_train_y, epochs=10, batch_size=32, validation_split=0.1, verbose=1)

            # -----------------------------------------------------------------------------------------------------

            fold_test_x, fold_test_y = sliding_window(fold_test_PPG, fold_test_ABP, sliding_window_width)
            fold_test_y = fold_test_y.reshape((fold_test_y.shape[0], fold_test_y.shape[1], 1))
            predict = model.predict(fold_test_x)

            rsme_predict = []
            for i in range(len(predict)):
                rsme_predict.append(predict[i][0])
            for i in range(sliding_window_width):
                rsme_predict.append(predict[len(predict) - 1][i])
            rsme_predict = scaler_ABP.inverse_transform(np.array(rsme_predict)).flatten()

            rsme_actual = []
            for i in range(len(fold_test_y)):
                rsme_actual.append(fold_test_y[i][0])
            for i in range(sliding_window_width):
                rsme_actual.append(fold_test_y[len(fold_test_y) - 1][i])
            rsme_actual = scaler_ABP.inverse_transform(np.array(rsme_actual)).flatten()

            rmse = math.sqrt(skm.mean_squared_error(rsme_actual, rsme_predict))
            if rmse < rmse_last:
                rmse_last = rmse
                model.save('/home/Data/model_%d.h5' % sliding_window_width)
