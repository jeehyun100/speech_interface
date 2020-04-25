import glob
import os
from sklearn import preprocessing
import numpy as np
import pandas as pd

def preprocess_ai_class_data():

    train_list = glob.glob('../SpeakerDB/Train/*_train.wav')
    test_list = glob.glob('../SpeakerDB/Test/*_test.wav')
    train_speaker_list = [(train_list[i].split('/')[-1].split('\\')[-1].split('_')[0],train_list[i].replace('..',''))  for i in range(len(train_list))]
    test_speaker_list = [(train_list[i].split('/')[-1].split('\\')[-1].split('_')[0], train_list[i].replace('..','')) for i in
                          range(len(test_list))]
    # All id collect train and test data
    np_train_speaker_list = np.array(train_speaker_list)
    np_test_speaker_list = np.array(test_speaker_list)
    np_all_speaker_list = np.vstack([np_train_speaker_list, np_test_speaker_list])

    # Make label encoder
    le = preprocessing.LabelEncoder()
    all_label_list = sorted(set(np_all_speaker_list[:,0]))
    print(all_label_list)
    le.fit(all_label_list)

    # To pandas
    column_list = ['id', 'path']
    df_train_prep_data = pd.DataFrame(np_train_speaker_list, columns=column_list)
    df_train_prep_data['encoded_id'] = le.transform(list(np_train_speaker_list[:,0]))

    df_test_prep_data = pd.DataFrame(np_test_speaker_list, columns=column_list)
    df_test_prep_data['encoded_id'] = le.transform(list(np_test_speaker_list[:,0]))

    # Save
    df_train_prep_data.to_csv('sic_train.txt', sep=' ', columns=['encoded_id', 'path'],#columns=['id', 'path', 'encoded_id'],
                     index=False, header=False)  # do not write index
    df_train_prep_data.to_csv('sic_test.txt', sep=' ', columns=['encoded_id', 'path'],
                     index=False, header=False)  # do not write index

if __name__ == '__main__':
    preprocess_ai_class_data()
    ## Encoding the variable
#fit = df.apply(lambda x: d[x.name].fit_transform(x))

# Inverse the encoded
#fit.apply(lambda x: d[x.name].inverse_transform(x))

# Using the dictionary to label future data
#df.apply(lambda x: d[x.name].transform(x))
