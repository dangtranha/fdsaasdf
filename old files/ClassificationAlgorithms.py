import pickle
import inspect
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from keras.layers import LSTM, Dense,TimeDistributed, Bidirectional, Flatten
from keras.layers import BatchNormalization
from sklearn.model_selection import KFold 

def random_forest(train_X, train_y, test_X, n_estimators=10, min_samples_leaf=5, criterion='entropy', print_model_details=False, gridsearch=True, save_model=False):

    if gridsearch:
        tuned_parameters = [{'min_samples_leaf': [2, 10, 50, 100, 200],
                             'n_estimators':[10, 50, 100],
                             'criterion':['gini', 'entropy']}]
        rf = GridSearchCV(RandomForestClassifier(), tuned_parameters, cv=5, scoring='accuracy', error_score= 'raise')
    else:
        rf = RandomForestClassifier(n_estimators=n_estimators, min_samples_leaf=min_samples_leaf, criterion=criterion)

    # Fit the model

    rf.fit(train_X, train_y.values.ravel())

    if gridsearch and print_model_details:
        print(rf.best_params_)

    if gridsearch:
        rf = rf.best_estimator_

    pred_prob_training_y = rf.predict_proba(train_X)
    pred_prob_test_y = rf.predict_proba(test_X)
    pred_training_y = rf.predict(train_X)
    pred_test_y = rf.predict(test_X)
    frame_prob_training_y = pd.DataFrame(pred_prob_training_y, columns=rf.classes_)
    frame_prob_test_y = pd.DataFrame(pred_prob_test_y, columns=rf.classes_)

    if print_model_details:
        ordered_indices = [i[0] for i in sorted(enumerate(rf.feature_importances_), key=lambda x:x[1], reverse=True)]
        print('Top 20 feature importances random forest:')
        for i in range(0, 20):
            print(train_X.columns[ordered_indices[i]], end='')
            print(' & ', end='')
            print(rf.feature_importances_[ordered_indices[i]])
    
    if save_model:
        # save the model to disk
        filename = 'final_' + str(inspect.stack()[0][3]) + '_model_BCI.sav'
        pickle.dump(rf, open(filename, 'wb'))

    return pred_training_y, pred_test_y, frame_prob_training_y, frame_prob_test_y

def lstm(X_train_input, y_train_input, X_test_input, y_test_input):
    X_train, y_train = X_train_input[:4000], y_train_input[:4000]
    X_test, y_test = X_test_input[:1000], y_test_input[:1000]

    X_train = X_train.to_numpy()
    y_train = y_train.to_numpy()
    X_train = X_train.reshape(4,1000,584)
    y_train = y_train.reshape(4,1000,1)
    print(np.shape(X_train),np.shape(X_test))

    model = Sequential()

    model.add(BatchNormalization(input_shape=(1000,584)))
    model.add((LSTM(64,input_shape=(1000, 584), activation='tanh',return_sequences=True)))
    # model.add(LSTM(units=128))
    model.add(TimeDistributed(Dense(1,activation='sigmoid')))

    model.compile(optimizer='Adam',loss='binary_crossentropy',metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=50, batch_size=1000, verbose=1)\
    
    class_test_y = (model.predict(X_test) > 0.5).astype("int32")
    class_test_y = class_test_y.reshape(1000,1)

    return class_test_y