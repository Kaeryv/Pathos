#!/bin/python3

import pandas as pd
import json
import os
import argparse

# Computer vision
import cv2
import dlib
import numpy as np



def detect_faces(detector, raw_image):

    # Convert the image to grayscale
    gray = cv2.cvtColor(raw_image, cv2.COLOR_BGR2GRAY)
    # We use dlib's face detector
    face_areas = detector(gray, 1)
    
    faces = list()

    # We detect all the faces in the picture

    for (i, area) in enumerate(face_areas):
        x1, x2, y1, y2 = area.tl_corner().x, area.br_corner().x, area.tl_corner().y, area.br_corner().y
        grayface = gray[y1:y2, x1:x2]
        faces.append(grayface)

    return faces

def g48_standardize(gray_image):
    gray_image = cv2.equalizeHist(gray_image)

    try:
        gray_image = cv2.resize(gray_image, (48, 48), interpolation=cv2.INTER_CUBIC) / 255.
    except Exception:
        print("[Error] While resizing.")
        return None
    return gray_image


def main(config_file, force, starting_index):
    if args.start_at is not None:
        start_at = args.start_at
    else:
        start_at = 0
    
    hog_fd = dlib.get_frontal_face_detector()

    # We start by loading the installation configuration
    configuration = json.load(config_file)
   
    # We get the variables from system installation
    database_root =         configuration['affectnet'][args.branch]['database_root']
    metafile =              configuration['affectnet'][args.branch]["meta"][args.metafile]
    emotion_map =           configuration['affectnet']['emotions_mapping']
    processed_foldername =  configuration['affectnet']['processed_foldername'] 
    raw_foldername =        configuration['affectnet']['raw_foldername'] 


    output_folder =  os.path.join(database_root, processed_foldername)
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
 
    def process_db(data):
        for index, row in data.iterrows():
            emotion_id = row[6]
            image_path = row[0]
            
            image_abs_path = os.path.join(database_root, raw_foldername)
            image_abs_path = os.path.join(image_abs_path, image_path)

            current_folder = os.path.join(output_folder, os.path.dirname(image_path))

      
            if not index%1000: 
                print('Progress: ' + str(index) + ' images.')
           
            # The file name without the extension
            filename = os.path.basename(image_abs_path)
            filename = os.path.splitext(filename)[0]
            
            # The output file
            processed_filename = os.path.join(current_folder, filename + '.png')

            if (os.path.isfile(processed_filename) and not force) or index < start_at:
                # In overwrite mode, we overwrite the existing processed image.
                # If condition is met, we skip this iteration.
                continue

            
            os.makedirs(current_folder, exist_ok=True)

            with cv2.imread(image_abs_path) as image_raw:
                faces = detect_faces(hog_fd, image_raw)

                if len(faces) != 0 and faces[0] is not None:
                    current_folder = os.path.join(output_folder, os.path.dirname(image_path))
                    os.makedirs(current_folder, exist_ok=True)
            
                    if not index%1000: 
                        print('Progress: ' + str(index) + ' images.')
                
                    # The file name without the extension
                    filename = os.path.splitext(os.path.basename(image_abs_path))[0]
                    
                    # The output file
                    processed_filename = os.path.join(current_folder, filename + '.png')

                    # In overwrite mode, we overwrite the existing processed image.
                    if os.path.isfile(processed_filename) and not force:
                        continue
                    
                    faces[0] = g48_standardize(faces[0])
                    if faces[0] is None:
                        continue

                    faces[0] = faces[0] * 255
                    faces[0] = faces[0].astype('uint8')
                    print("Processed " + image_path)
                    cv2.imwrite(processed_filename, faces[0])
        
    metapath = os.path.join(database_root, metafile)
    print("[INFO] Loading metadata at " + metapath)
    data = pd.read_csv(metapath)
    print("[INFO] Starting conversion")
    process_db(data)

    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Preprocesses affectnet database to \
            generate standardized dataset suited for Pathos.')
    parser.add_argument('-c', '--config', metavar='config', required=True, help="Configuration file for the Database.", type=argparse.FileType('r'))
    parser.add_argument('-f', '--force', action='store_true')
    parser.add_argument('-s', '--starting-index', dest='start_at', default=0)
    parser.add_argument('--database-branch', dest='branch')
    args = parser.parse_args()

    main(args.config, args.force, args.start_at)
