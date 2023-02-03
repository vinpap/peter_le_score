# -*- coding: utf-8 -*-
"""
Example script

Script to perform some corrections in the brief audio project

Created on Fri Jan 27 09:08:40 2023

@author: ValBaron10
"""

# Import
import time
import pickle
from pathlib import Path
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from features_functions import compute_features

from sklearn import preprocessing
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import accuracy_score

def filter_features(features, features_selector):
    """
    Enables us to decide which features should be used.
    features_selector is an array containing boolean values
    that is used to define what features (columns in the 
    'features' array) should be used.
    """

    features = features[:, features_selector]

    return features

def train_model(select_features=False, features_selector=[0]):
    """
    Convenience function for automated model testing. Parameters:
    - select_features : if set to False, all features are used. Otherwise,
    we only use the features as defined by features_selector
    - features_selector : only works if select_features is set at True. 
    It is an array-like object of length 71 (the max number of features) containing
    1s and 0s
    """

    # Set the paths to the files 
    data_path = "Data/"


    # In order to save time, we don't recompute the features every time
    # the function is called. Instead, we pickle them.

    features_path = Path('./features.pkl')
    labels_path = Path('./labels.pkl')

    if not (features_path.is_file() and labels_path.is_file()):

        # Names of the classes
        classes_paths = ["Cars/", "Trucks/"]
        classes_names = ["car", "truck"]
        cars_list = [4,5,7,9,10,15,20,21,23,26,30,38,39,44,46,48,51,52,53,57]
        trucks_list = [2,4,10,11,13,20,22,25,27,30,31,32,33,35,36,39,40,45,47,48]
        nbr_of_sigs = 20 # Nbr of sigs in each class
        seq_length = 0.2 # Nbr of second of signal for one sequence
        nbr_of_obs = int(nbr_of_sigs*10/seq_length) # Each signal is 10 s long

        print("COMPUTING FEATURES...")
        t1 = time.time()

        # Go to search for the files
        learning_labels = []
        for i in range(2*nbr_of_sigs):
            if i < nbr_of_sigs:
                name = f"{classes_names[0]}{cars_list[i]}.wav"
                class_path = classes_paths[0]
            else:
                name = f"{classes_names[1]}{trucks_list[i - nbr_of_sigs]}.wav"
                class_path = classes_paths[1]

            # Read the data and scale them between -1 and 1
            fs, data = sio.wavfile.read(data_path + class_path + name)
            data = data.astype(float)
            data = data/32768

            # Cut the data into sequences (we take off the last bits)
            data_length = data.shape[0]
            nbr_blocks = int((data_length/fs)/seq_length)
            seqs = data[:int(nbr_blocks*seq_length*fs)].reshape((nbr_blocks, int(seq_length*fs)))

            for k_seq, seq in enumerate(seqs):
                # Compute the signal in three domains
                sig_sq = seq**2
                sig_t = seq / np.sqrt(sig_sq.sum())
                sig_f = np.absolute(np.fft.fft(sig_t))
                sig_c = np.absolute(np.fft.fft(sig_f))

                # Compute the features and store them
                features_list = []
                N_feat, features_list = compute_features(sig_t, sig_f[:sig_t.shape[0]//2], sig_c[:sig_t.shape[0]//2], fs)
                features_vector = np.array(features_list)[np.newaxis,:]

                if k_seq == 0 and i == 0:
                    learning_features = features_vector
                    learning_labels.append(classes_names[0])
                elif i < nbr_of_sigs:
                    learning_features = np.vstack((learning_features, features_vector))
                    learning_labels.append(classes_names[0])
                else:
                    learning_features = np.vstack((learning_features, features_vector))
                    learning_labels.append(classes_names[1])

        with open("features.pkl", "wb") as features_file: 
            pickle.dump(learning_features, features_file)
        with open("labels.pkl", "wb") as labels_file: 
            pickle.dump(learning_labels, labels_file)

        # print(learning_features.shape)
        # print(len(learning_labels))

        t2 = time.time()
        print(f"FEATURES COMPUTED IN {t2-t1}s")

    else:
        with open("features.pkl", "rb") as features_file: 
            learning_features = pickle.load(features_file)
        with open("labels.pkl", "rb") as labels_file: 
            learning_labels = pickle.load(labels_file)

    if select_features: learning_features = filter_features(learning_features, features_selector)

    print("TRAINING MODEL...")
    t3 = time.time()

    # Separate data in train and test
    X_train, X_test, y_train, y_test = train_test_split(learning_features, learning_labels, test_size=0.2, random_state=42)

    # Standardize the labels
    labelEncoder = preprocessing.LabelEncoder().fit(y_train)
    learningLabelsStd = labelEncoder.transform(y_train)
    testLabelsStd = labelEncoder.transform(y_test)

    # Learn the model
    model = svm.SVC(C=10, kernel='linear', class_weight=None, probability=False)
    scaler = preprocessing.StandardScaler(with_mean=True).fit(X_train)
    learningFeatures_scaled = scaler.transform(X_train)

    model.fit(learningFeatures_scaled, learningLabelsStd)

    t4 = time.time()
    print(f"MODEL TRAINED IN {t4-t3} s")

    # Test the model
    testFeatures_scaled = scaler.transform(X_test)
    predictedLabels = model.predict(testFeatures_scaled)
    accuracy = accuracy_score(testLabelsStd, predictedLabels)

    # Matrix confusion
    #plot_confusion_matrix(model, testFeatures_scaled, testLabelsStd) 
    #plt.show()

    results = {
                "model": "SVM",
                "accuracy": accuracy,
    }
    if select_features: results["features_used"] = features_selector
    else: results["features_used"] = np.array([1]*71)

    return results