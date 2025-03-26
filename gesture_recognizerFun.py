#REFERENCE: https://github.com/google/mediapipe/issues/4448

import multiprocessing
import sys

from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np
import cv2

# STEP 1: Import the necessary modules.
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

import argparse
import time
import socket,os,struct, time

# Definisci le costanti per la visualizzazione delle landmarks
MARGIN = 10  # pixel
FONT_SIZE = 1
FONT_THICKNESS = 1
HANDEDNESS_TEXT_COLOR = (88, 205, 54)  # verde vibrante

# 👍, 👎, ✌️, ☝️, ✊, 👋, 🤟 
BaseOptions = mp.tasks.BaseOptions
GestureRecognizer = mp.tasks.vision.GestureRecognizer
GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
GestureRecognizerResult = mp.tasks.vision.GestureRecognizerResult
VisionRunningMode = mp.tasks.vision.RunningMode

def gesture_recognizer(stream_queue, result_queue):


    # Visualization parameters
    row_size = 20  # pixels
    left_margin = 24  # pixels
    text_color = (0, 0, 255)  # red
    font_size = 1
    font_thickness = 1
    fps_avg_frame_count = 10

    counter, fps = 0,0
    start_time = time.time()

    mp_drawing = solutions.drawing_utils
    mp_pose = solutions.pose
    results = None

    # Aggiungi una funzione per verificare se una sequenza di gesture corrisponde a una sequenza preregistrata
    # def check_sequence_match(recorded_sequence):
    #     for sequence, command in command_sequences.items():
    #         if recorded_sequence == sequence:
    #             return command
    #     return None
    

    def draw_landmarks_on_image(rgb_image, detection_result):
        hand_landmarks_list = detection_result.hand_landmarks
        handedness_list = detection_result.handedness
        annotated_image = np.copy(rgb_image)

        # Loop through the detected hands to visualize.
        for idx in range(len(hand_landmarks_list)):
            hand_landmarks = hand_landmarks_list[idx]
            handedness = handedness_list[idx]

            # Draw the hand landmarks.
            hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
            hand_landmarks_proto.landmark.extend([
            landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in hand_landmarks
            ])
            solutions.drawing_utils.draw_landmarks(
            annotated_image,
            hand_landmarks_proto,
            solutions.hands.HAND_CONNECTIONS,
            solutions.drawing_styles.get_default_hand_landmarks_style(),
            solutions.drawing_styles.get_default_hand_connections_style())

            # Get the top left corner of the detected hand's bounding box.
            height, width, _ = annotated_image.shape
            x_coordinates = [landmark.x for landmark in hand_landmarks]
            y_coordinates = [landmark.y for landmark in hand_landmarks]
            text_x = int(min(x_coordinates) * width)
            text_y = int(min(y_coordinates) * height) - MARGIN

            # Draw handedness (left or right hand) on the image.
            cv2.putText(annotated_image, f"{handedness[0].category_name}",
                        (text_x, text_y), cv2.FONT_HERSHEY_DUPLEX,
                        FONT_SIZE, HANDEDNESS_TEXT_COLOR, FONT_THICKNESS, cv2.LINE_AA)

        return annotated_image
    # Create a pose landmarker instance with the live stream mode:
    def print_result(result: GestureRecognizerResult, output_image: mp.Image, timestamp_ms: int):
        #print('pose landmarker result: {}'.format(result))
        nonlocal results
        results = result
        #print(type(result))
        
    def external_stream(self,frame):
        cv2.imshow('Color', frame)


    current_pid = os.getpid()
    print("*********************PID del processo gesture_recognizerModule in run:", current_pid, '*********************')


    options = GestureRecognizerOptions(
            base_options=BaseOptions(model_asset_path='/home/francesco/mp-models/gesture_recognizer.task', delegate="GPU"),
            running_mode=VisionRunningMode.LIVE_STREAM,
            result_callback=print_result)

    timestamp = 0
    with GestureRecognizer.create_from_options(options) as landmarker:
        # The landmarker is initialized. Use it here.
        # ...
        i = 0
        while(1):
            #print('gestureModule: looping', i)
            i+=1
            
            timestamp += 1
            counter += 1
            color_img = stream_queue.get()
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=color_img)
            landmarker.recognize_async(mp_image, timestamp)

                                # Calculate the FPS
            if counter % fps_avg_frame_count == 0:
                end_time = time.time()
                fps = fps_avg_frame_count / (end_time - start_time)
                start_time = time.time()

            current_frame = mp_image.numpy_view()

            # Show the FPS
            fps_text = 'FPS = {:.1f}'.format(fps)
            text_location = (left_margin, row_size)

            #print(fps_text)
            #print(results)

            if(not (results is None)):
                annotated_image = draw_landmarks_on_image(current_frame, results)
                #cv2.imshow('Show',cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))

                cv2.putText(annotated_image, fps_text, text_location, cv2.FONT_HERSHEY_PLAIN,
                font_size, text_color, font_thickness)
                
                cv2.namedWindow('Detected Landmarks', cv2.WINDOW_NORMAL)
                cv2.resizeWindow('Detected Landmarks', 800, 600)
                cv2.imshow('Detected Landmarks',annotated_image)
                #print(self.results.gestures)   

                if results.gestures:
                    for gesture_list in results.gestures:
                    # Trova il gesto con lo score più alto
                        top_gesture = max(gesture_list, key=lambda gesture: gesture.score)

                        #print(top_gesture.category_name)
                        result_queue.put(top_gesture.category_name)
                        #event.set()
                        # print(f'result_gesture_queue: {result_queue.qsize()}')

            else:
                cv2.imshow('Self result is null', color_img)

            if cv2.waitKey(5) & 0xFF == ord('q'):
                print("Closing Camera Stream")
                sys.exit(' Termino processo Mediapipe Gesture')

            cv2.waitKey(1)