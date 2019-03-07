from os.path import join
import os
import numpy as np
from pathos.constants import *
from sklearn.model_selection import train_test_split

ancm = 'both'
class DatasetLoader:
    def load_from_save(self):

        print("___________________________________________________________________")
        print("-------------------------------------------------------------------")
        print("_________________________ P.A.T.H.O.S LEARN _______________________")
        print("----------------------------DATASETLOADER--------------------------")
        print("___________________________________________________________________")

        path_images = join(SAVE_DIRECTORY, 'affectnet.images.' + ancm + '.npy')
        path_labels = join(SAVE_DIRECTORY, 'affectnet.labels.' + ancm + '.npy')

        print("[INFO] Loading numpy database")
        print('[INFO] IMAGES Database location : ' + path_images)
        print('[INFO] LABELS Database location : ' + path_labels)
        
        if(os.path.isfile(path_images)):
            print("[INFO] IMAGES Database was found.") 
        if(os.path.isfile(path_labels)):
            print("[INFO] LABELS Database was found.") 

        self._labels = np.load(path_labels)
        print("[INFO] LABELS memory [" + str(self._labels.nbytes//2**20) + " Mb].") 
        
        self._images = np.load(path_images)
        print("[INFO] IMAGES memory [" + "{:.2f}".format(self._images.nbytes//(2**30)) + " Gb].")


        print("[INFO] Reshaping LABELS")
        self._labels = self._labels.reshape([-1, len(EMOTIONS)])
        
        print("[INFO] Reshaping IMAGES")
        self._images = self._images.reshape([-1, SENSOR_SIZE, SENSOR_SIZE, 1])

        
        print("[INFO] Splitting training and validation samples")
        print(str(self._images.shape[0]) + ' samples :')
        self._images, self._images_test, self._labels, self._labels_test = \
            train_test_split(self._images, \
            self._labels, \
            test_size=0.20, random_state=42)
        
        print('\t-' + str(self._images.shape[0]) + ' samples for training.')
        print('\t-' + str(self._images_test.shape[0]) + ' samples for testing.')

    @property
    def images(self):
        return self._images

    @property
    def labels(self):
        return self._labels

    @property
    def images_test(self):
        return self._images_test

    @property
    def labels_test(self):
        return self._labels_test
