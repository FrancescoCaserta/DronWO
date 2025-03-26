import multiprocessing
import time
import opencvViewerFun
import gesture_recognizerFun
import object_recognizerFun
import crazyflieProcessFun

if __name__ == '__main__':
    multiprocessing.set_start_method('spawn')

    stream1_queue = multiprocessing.Queue()
    stream2_queue = multiprocessing.Queue()

    result_gesture_queue = multiprocessing.Queue()
    result_object_queue = multiprocessing.Queue()

    #new_gesture_event = multiprocessing.Event()


    viewer_process = multiprocessing.Process(target=opencvViewerFun.viewerFunction, args =(stream1_queue,stream2_queue))
    viewer_process.start()
    time.sleep(3)

    gesture_process = multiprocessing.Process(target=gesture_recognizerFun.gesture_recognizer, args=(stream1_queue,result_gesture_queue,)) #new_gesture_event
    gesture_process.start()


    obj_detection_event = multiprocessing.Event()
    object_process = multiprocessing.Process(target=object_recognizerFun.object_recognizer, args=(stream2_queue,result_object_queue, obj_detection_event))
    object_process.start()

    time.sleep(10)
    cf_process = multiprocessing.Process(target=crazyflieProcessFun.command_crazyflie, args=(result_gesture_queue, result_object_queue, obj_detection_event)) #result_gesture_queue, new_gesture_event
    cf_process.start()


    print(' here')
    cf_process.join()
    object_process.join()
    gesture_process.join()
    viewer_process.join() #if i dont write this, the shared queue is deleted by the GarbageCollector and the other process has an error.
    