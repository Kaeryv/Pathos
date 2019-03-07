#!/bin/python3

import pandas as pd
import json
import os
import argparse

# Computer vision
import cv2
import dlib
import numpy as np
import constants


def emotion_to_vec(x):
    d = np.zeros(len(constants.EMOTIONS))
    d[x] = 1.0
    return d


if __name__ == '__main__':
    print('starting preprocess.')

    parser = argparse.ArgumentParser(description='Stores affectnet preprocessed database in numpy cache')
    parser.add_argument('--training-database', dest='training', action='store_true')
    parser.add_argument('--validation-database', dest='validation', action='store_true')
    parser.add_argument('--both-database', dest='both', action='store_true')
    args = parser.parse_args()



    # We start by loading the installation configuration
    f = open('./config.json')
    configuration = json.load(f)
    f.close()
    database_root = configuration['affectnet']['database_root']
    training_meta = configuration['affectnet']['training_metafile']
    validation_meta = configuration['affectnet']['validation_metafile']
    emotion_map = configuration['affectnet']['emotions_mapping']
    preprocess_folder = configuration['affectnet']['processed_foldername']
    processed_files_folder = os.path.join(database_root, preprocess_folder) 

    images = list()    
    labels = list()

    def process_db(data, name):
        if not os.path.exists(processed_files_folder):
            print('First, process the raw databae first')
            exit()
        for index, row in data.iterrows():
            emotion_id = row[6]
            fer_emotion_id = configuration['affectnet']['affectnet2fer'][emotion_id]
            image_path = row[0]
            current_folder = os.path.join(processed_files_folder, os.path.dirname(image_path))
            filename = os.path.basename(image_path)
            filename = os.path.splitext(filename)[0]
            processed_filename = os.path.join(current_folder, filename + '.png')
            
            if os.path.isfile(processed_filename) and fer_emotion_id < 7:
                processed_image = cv2.imread(processed_filename)
                processed_image = cv2.cvtColor(processed_image, cv2.COLOR_BGR2GRAY)
                print(str(index) + " - " + processed_filename)
                labels.append(emotion_to_vec(fer_emotion_id))
                images.append(processed_image / 255)

        np.save('affectnet.pictures.' + name , images)
        np.save('affectnet.labels.' + name, labels)
    if args.training:
        data = pd.read_csv(os.path.join(database_root, training_meta))
        process_db(data, 'training')
    if args.validation:
        data = pd.read_csv(os.path.join(database_root, validation_meta))
        process_db(data, 'validation')
    if args.both:
        data = pd.read_csv(os.path.join(database_root, validation_meta))
        process_db(data, 'both')
        data = pd.read_csv(os.path.join(database_root, training_meta))
        process_db(data, 'both')
