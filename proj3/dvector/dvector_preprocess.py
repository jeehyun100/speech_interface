import glob
import os
from sklearn import preprocessing
import numpy as np
import pandas as pd
import librosa
from AudioAugmentation import AudioAugmentation
from shutil import copyfile


def preprocess_ai_class_data(filepath):

    train_list = glob.glob(filepath + 'Train/*_train.wav')
    test_list = glob.glob(filepath + 'Test/*_test.wav')
    train_speaker_list = [(train_list[i].split('/')[-1].split('\\')[-1].split('_')[0],train_list[i].replace('..',''))  for i in range(len(train_list))]
    test_speaker_list = [(test_list[i].split('/')[-1].split('\\')[-1].split('_')[0], test_list[i].replace('..','')) for i in
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
    df_test_prep_data.to_csv('sic_test.txt', sep=' ', columns=['encoded_id', 'path'],
                     index=False, header=False)  # do not write index

def split_5sec(filepath, fs = 16000):
    aug_dir = "../SpeakerAugDB/"
    aug_dir_train = "../SpeakerAugDB/Train/"
    aug_dir_test = "../SpeakerAugDB/Test/"
    os.makedirs(aug_dir_train, exist_ok=True)
    os.makedirs(aug_dir_test, exist_ok=True)

    train_list = glob.glob(filepath+'Train/*_train.wav')
    #test_list = glob.glob('../SpeakerDB/Test/*_test.wav')
    for data_path in train_list:
        sig, fs = librosa.load(data_path, sr=fs)

        # how many split files?
        n_chunk = int((len(sig) / fs) / 3.0)
        sig_n_chunk_array = np.array_split(sig, n_chunk)
        save_data_filename = os.path.splitext(os.path.basename(data_path))[0]
        save_data_path = os.path.dirname(data_path)
        for i, split_array in enumerate(sig_n_chunk_array):
            split_filename = save_data_filename.split("_")[0] + "_" +str(i) + "_" + save_data_filename.split("_")[1] + ".wav"
            save_path = aug_dir_train + split_filename

            librosa.output.write_wav(save_path, split_array, fs)
            print("Save file : {0}".format(save_path))

        print("{0} wav length {1} s".format(data_path, len(sig)/fs))

    #     #y, sr = librosa.load(wav, sr=16000)
    #     time = np.linspace(0, len(y) / sr, len(y))  # time axis
    # test_speaker_list = [(test_list[i].split('/')[-1].split('\\')[-1].split('_')[0], test_list[i].replace('..', '')) for
    #                      i in
    #                      range(len(test_list))]

def audio_aug(aug_dir):
    #aug_dir = "../SpeakerAugDB/"
    aug_dir_train = "../SpeakerAug2DB/Train/"
    aug_dir_test = "../SpeakerAug2DB/Test/"
    os.makedirs(aug_dir_train, exist_ok=True)
    os.makedirs(aug_dir_test, exist_ok=True)

    train_list = glob.glob(aug_dir+'Train/*_train.wav')
    #test_list = glob.glob('../SpeakerDB/Test/*_test.wav')

    aa = AudioAugmentation()
    for data_path in train_list:
        save_data_filename = os.path.splitext(os.path.basename(data_path))[0]
        data = aa.read_audio_file(data_path)
        # Adding noise to sound
        data_noise = aa.add_noise(data)
        # Shifting the sound
        data_roll = aa.shift(data)
        # Stretching the sound
        data_stretch = aa.stretch(data, 0.8)
        # Write generated cat sounds
        split_filename = save_data_filename.split("_")[0] + "_noise_" + save_data_filename.split("_")[
            1] + "_" + save_data_filename.split("_")[2] + ".wav"
        save_path = aug_dir_train + split_filename
        aa.write_audio_file(save_path, data_noise)

        split_filename = save_data_filename.split("_")[0] + "_roll_" + save_data_filename.split("_")[
            1] + "_" + save_data_filename.split("_")[2] + ".wav"
        save_path = aug_dir_train + split_filename
        aa.write_audio_file(save_path, data_roll)

        split_filename = save_data_filename.split("_")[0] + "_stre_" + save_data_filename.split("_")[
            1] + "_" + save_data_filename.split("_")[2] + ".wav"
        save_path = aug_dir_train + split_filename
        #aa.write_audio_file(save_path, data_roll)
        aa.write_audio_file(save_path, data_stretch)

        split_filename = save_data_filename.split("_")[0] + "_ori_" + save_data_filename.split("_")[
            1] + "_" + save_data_filename.split("_")[2] + ".wav"
        dst = aug_dir_train + split_filename
        copyfile(data_path, dst)
        print("Augmentation file [noise, roll. stretch]{0}".format(data_path))




if __name__ == '__main__':
    filepath = "../modify_data/"
    #filepath = "../SpeakerAug2DB/"
    #preprocess_ai_class_data(filepath)
    #split_5sec(filepath)
    #filepath = "../SpeakerAugDB/"
    audio_aug(filepath)

