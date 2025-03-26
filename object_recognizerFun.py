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

MARGIN = 10  # pixels
ROW_SIZE = 10  # pixels
FONT_SIZE = 1
FONT_THICKNESS = 1
TEXT_COLOR = (255, 0, 0)  # red


def object_recognizer(stream_queue, result_queue, obj_detection_event):


    # Visualization parameters
    row_size = 20  # pixels
    left_margin = 24  # pixels
    text_color = (0, 0, 255)  # red
    font_size = 1
    font_thickness = 1
    fps_avg_frame_count = 10

    latest_detection_result = None

    counter, fps = 0,0
    start_time = time.time()

    # Aggiungi una funzione per verificare se una sequenza di gesture corrisponde a una sequenza preregistrata
    # def check_sequence_match(recorded_sequence):
    #     for sequence, command in command_sequences.items():
    #         if recorded_sequence == sequence:
    #             return command
    #     return None
    


    def visualize(
            image,
            detection_result
        ) -> np.ndarray:
        """Draws bounding boxes on the input image and return it.
        Args:
            image: The input RGB image.
            detection_result: The list of all "Detection" entities to be visualized.
        Returns:
            Image with bounding boxes.
        """
        for detection in detection_result.detections:
            # Draw bounding_box
            bbox = detection.bounding_box
            start_point = bbox.origin_x, bbox.origin_y
            end_point = bbox.origin_x + bbox.width, bbox.origin_y + bbox.height
            cv2.rectangle(image, start_point, end_point, TEXT_COLOR, 3)

            # Draw label and score
            category = detection.categories[0]
            category_name = category.category_name
            probability = round(category.score, 2)
            result_text = category_name + ' (' + str(probability) + ')'
            text_location = (MARGIN + bbox.origin_x,
                            MARGIN + ROW_SIZE + bbox.origin_y)
            cv2.putText(image, result_text, text_location, cv2.FONT_HERSHEY_PLAIN,
                        FONT_SIZE, TEXT_COLOR, FONT_THICKNESS)

        return image
    
    def visualize_callback(result: vision.ObjectDetectorResult,
                        output_image: mp.Image, timestamp_ms: int):
        nonlocal latest_detection_result
        latest_detection_result = result


    current_pid = os.getpid()
    print("*********************PID del processo object_recognizerFun in run:", current_pid, '*********************')

    # Initialize the object detection model
    base_options = python.BaseOptions(model_asset_path='/home/francesco/mp-models/ssd_mobilenet_v2_int8.tflite', delegate="GPU")
    options = vision.ObjectDetectorOptions(base_options=base_options,
                                            running_mode=vision.RunningMode.LIVE_STREAM,
                                            score_threshold=0.5,
                                            result_callback=visualize_callback)
    detector = vision.ObjectDetector.create_from_options(options)


    frame_delay = int(1000 / 15)  # Calculate delay in milliseconds. 15 FPS (change here the fps desired)

    timestamp = 0
    with vision.ObjectDetector.create_from_options(options) as detector:
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

            detector.detect_async(mp_image, counter)

                # Calculate the FPS
            if counter % fps_avg_frame_count == 0:
                end_time = time.time()
                fps = fps_avg_frame_count / (end_time - start_time)
                start_time = time.time()

            current_frame = mp_image.numpy_view()
            current_frame = cv2.cvtColor(current_frame, cv2.COLOR_RGB2BGR) 

            # Show the FPS
            fps_text = 'FPS = {:.1f}'.format(fps)
            text_location = (left_margin, row_size)

            #print(fps_text)
            #print(results)
            #print('################')
            if obj_detection_event.is_set():
                if(not (latest_detection_result is None)):
                    #taking only the category_name of each object detected to be printed
                    class_names = []
                    for detection in latest_detection_result.detections: #latest_detection_result.detections contains all detections in a given frame.
                        #print('one detection')
                        #print(detection.__dict__) #{'bounding_box': BoundingBox(origin_x=226, origin_y=112, width=99, height=130), 'categories': [Category(index=None, score=0.6814852952957153, display_name=None, category_name='chair')], 'keypoints': []}
                        ################
                        # one detection
                        # {'bounding_box': BoundingBox(origin_x=153, origin_y=22, width=93, height=64), 'categories': [Category(index=None, score=0.6002792716026306, display_name=None, category_name='tv')], 'keypoints': []}
                        # one detection
                        # {'bounding_box': BoundingBox(origin_x=1, origin_y=14, width=155, height=136), 'categories': [Category(index=None, score=0.5515367984771729, display_name=None, category_name='tv')], 'keypoints': []}
                        # one detection
                        # {'bounding_box': BoundingBox(origin_x=78, origin_y=162, width=139, height=61), 'categories': [Category(index=None, score=0.5288079380989075, display_name=None, category_name='keyboard')], 'keypoints': []}
                        # ################
                        result_queue.put(detection)
                    #     for category in detection.categories: #why this? it seems that categories has always only one object detected.
                    #         class_names.append(category.category_name)
                    # if len(class_names) != 0:
                    #     for gesture in class_names:
                    #         result_queue.put(gesture)


                    annotated_image = visualize(current_frame, latest_detection_result)
                    #cv2.imshow('Show',cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))

                    cv2.putText(annotated_image, fps_text, text_location, cv2.FONT_HERSHEY_PLAIN,
                    font_size, text_color, font_thickness)
                    
                    cv2.namedWindow('Detected Objects', cv2.WINDOW_NORMAL)
                    cv2.resizeWindow('Detected Objects', 800, 600)
                    cv2.imshow('Detected Objects',annotated_image)
                    #print(self.results.gestures)   
                    latest_detection_result = None

                else:
                    cv2.imshow('Self result is null', color_img)

            if cv2.waitKey(5) & 0xFF == ord('q'):
                print("Closing Camera Stream")
                sys.exit(' Termino processo Mediapipe Object')

            cv2.waitKey(1)