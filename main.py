#!/usr/bin/env python3

'''
 Main

  Command-line interface of P.A.T.H.O.S.

 Author

  Nicolas Roy (UNamur)
  Travail effectué au département de psychologie de l'UNamur (2018)

 Usage

  The application is operated from command line.
  On UNIX systems;

     $ python main.py 
    
    or 
    
     $ ./main.py if main.py is marked as executable.
    
    To mark main.py as executable, use 
     
     $ chmod u+x main.py
  
  On windows, use

    python main.py


Entrainement:
    python main.py --train --model test

'''


import os
import cv2
import numpy as np
from imutils import face_utils
import imutils
import dlib
import argparse

from pathos.constants import FACIAL_LANDMARKS_IDXS, EMOTIONS, COLORS
from pathos.kerasmodel import EmotionRecognitionModel
from pathos import utils as utils
from pathos.anatomy.eye import Eye

ui_scale = 1.0

def annotate_frame(frame):
    #ui_scale = (bgr_image.shape[1] / 1024.0)
    gray_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
    # detect faces in the grayscale image
    rects = detector(gray_image, 1)
    # loop over the face detections
    for (i, rect) in enumerate(rects):
        x1, x2, y1, y2 = rect.tl_corner().x, rect.br_corner().x, rect.tl_corner().y, rect.br_corner().y
        face_coordinates = x1, y1, x2-x1, y2-y1
        
        
        focused_face_gray = gray_image[y1:y2, x1:x2]

        # Equalize histogram to get better contrast (span) 
        focused_face_gray = cv2.equalizeHist(focused_face_gray)
        # Format to g48
        if focused_face_gray is None:
            focused_face_gray = np.zeros((48, 48))
        focused_face_gray = cv2.resize(focused_face_gray, (48, 48), interpolation=cv2.INTER_CUBIC) / 255.

        '''
            Display a little inset of the sensor input on top left corner
        '''
        
        '''
        focused_face_display = focused_face_gray * 255 
        focused_face_display = focused_face_display.astype('uint8')
        display_focused = cv2.cvtColor(focused_face_display, cv2.COLOR_GRAY2BGR)
        bgr_image[0:48,0:48,:] = display_focused
        '''
        emotions_weights = emotion_network.predict(focused_face_gray)
        utils.draw_bounding_box((rect.left(), rect.top(), rect.width(), rect.height()), bgr_image, (0, 0, 0))
        if emotions_weights is not None:
            emotion_probability = np.max(emotions_weights)
            emotion_label_arg = np.argmax(emotions_weights)
            
            color = emotion_probability * np.asarray(COLORS[emotion_label_arg])
            for (proba, emotion, index) in zip(emotions_weights[0], EMOTIONS, range(len(EMOTIONS))):
                cv2.rectangle(bgr_image, (face_coordinates[0] + face_coordinates[2],
                    face_coordinates[1] + int(index * 30 * ui_scale)),
                                (face_coordinates[0] + face_coordinates[2] + int(proba * 100* ui_scale),
                                face_coordinates[1] + int((index + 1) * 30 * ui_scale)), (200, 200, 200, 200), -1)
                utils.draw_text(face_coordinates, bgr_image, emotion + " " + str(int(proba * 100)), color,
                            face_coordinates[2] + int(10 * ui_scale), int(index * 30 * ui_scale)
                            + int(18 * ui_scale), 0.5 * ui_scale, 1)

        shape = predictor(gray_image, rect)
        if shape is not None:
            shape = face_utils.shape_to_np(shape)
        
            # loop over the face parts individually
        
        
            for (name, (i, j)) in face_utils.FACIAL_LANDMARKS_IDXS.items():
    
                for (x, y) in shape[i:j]:
                    cv2.circle(bgr_image, (x, y), max(int(2 * ui_scale), 1), (0, 0, 255), -1)
    
                # Localisation du barycentre de la zone occulaire
                eye_pos_x = 0
                eye_pos_y = 0
                if name == "left_eye" or name == "right_eye":
                    for (x, y) in shape[i:j]:
                        eye_pos_x += x
                        eye_pos_y += y
                    eye_pos_x = eye_pos_x / (j-i)
                    eye_pos_y = eye_pos_y / (j-i)
    
                # if name == "right_eye":
                    right_eye = Eye(shape[i:j])
                    if right_eye.aspect_ratio() < 0.3:
                        pass
                    else:
                        cv2.circle(bgr_image, (int(eye_pos_x), int(eye_pos_y)), max(int(4 * ui_scale),
                            1), (255, 0, 0))
            if emotions_weights is not None:
                pass
                #return emotions_weights
            else:
                pass



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Annotates media with 7 emotions weights.')
    parser.add_argument('--image', '-i', dest="image", help='Any supported media.')
    parser.add_argument('--folder', '-f', dest="folder", help='Any supported media folder.')
    parser.add_argument('--video', '-v', dest="video", help='Any supported media.')
    parser.add_argument('--json', '-j', dest="json", help='Metadata.')
    parser.add_argument('--output', '-o', dest="output", help='Any supported media.')
    parser.add_argument('--model', '-m',dest="model", help='Any supported media.')
    parser.add_argument('--webcam','-w', action='store_true')
    parser.add_argument('--train','-t', action='store_true')
    args = parser.parse_args()
    #print(args.webcam)
    
    print('[Info] Loading dlib face detector') 
    detector = dlib.get_frontal_face_detector()
    
    print("[Info] Loading facial landmarks detector")
    predictor = dlib.shape_predictor("db/shape_predictor_68_face_landmarks.dat")
    
    print('[Info] Loading emotion recognition network') 

    if args.model is not None:
        emotion_network = EmotionRecognitionModel(args.model)
    else:
        emotion_network = EmotionRecognitionModel("default")
    
   
    if args.train:
        print('[Info] Started training model')
        emotion_network.train()
        emotion_network.save_model()
        exit()

    emotion_network.build_network()
    emotion_network.load_model()
    
    if args.webcam:
        print('[Info] Initializing video capture')
        cap = cv2.VideoCapture(0)
    
        print('[Info] Opening display.')

        #cv2.namedWindow('Probably Accurate TransAnthropomorphic Heuristic Optimisation  System', cv2.WINDOW_NORMAL)
        while cap.isOpened():

            # Capturing the image from webcam
            bgr_image = cap.read()[1]
        
            annotate_frame(bgr_image)
        
            cv2.imshow('Webcam', bgr_image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cv2.destroyAllWindows()
        cap.release()


    elif args.folder is not None:
        for root, dirs, files in os.walk(args.folder): 
            for image in files:
                print(image)
                bgr_image = cv2.imread(args.folder + image)
                annotate_frame(bgr_image)
                cv2.imwrite( args.folder + os.path.splitext(image)[0] + "." + args.model + ".png", bgr_image)
    elif args.image is not None and args.output is not None:
        print("anotating image")
        bgr_image = cv2.imread(args.image)
        annotate_frame(bgr_image)
        cv2.imwrite(args.output, bgr_image)

    elif args.video is not None and args.output is not None:
        print('[Info] Initializing video capture from file.')
        print(args.video)
        cap = cv2.VideoCapture("./" + args.video)
        print(args.video)
        out = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc(*'MPEG'), 20.0, (int(cap.get(3)),int(cap.get(4))))
        frames = -1
        while cap.isOpened() and out.isOpened():
            frames = frames + 1
            print("Processed frame " + str(frames))
            if frames < 1000:
                continue
            # Capturing the image from webcam
            bgr_image = cap.read()[1]
            annotate_frame(bgr_image)
            cv2.imshow(args.video, bgr_image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                out.write(bgr_image)
                cap.release()
                out.release()
                break
            out.write(bgr_image)
        cap.release()
        out.release()
