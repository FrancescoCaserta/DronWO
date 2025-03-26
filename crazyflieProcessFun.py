#REFERENCES:
# The trajectory to fly
# See https://github.com/whoenig/uav_trajectories for a tool to generate
# trajectories
#paper: https://groups.csail.mit.edu/rrg/papers/Richter_ISRR13.pdf


import csv
import math
import subprocess
import sys
import time
import os
#import threading
from queue import Queue, Empty
import multiprocessing


import cflib.crtp
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.log import LogConfig
from cflib.crazyflie.mem import MemoryElement
from cflib.crazyflie.mem import Poly4D
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie
from cflib.crazyflie.syncLogger import SyncLogger

from cflib.utils import uri_helper


from pynput import keyboard



# Flag per terminare il ciclo
running = True
FLIGHT_MODE = False #to test camera in on flight mode.
flying = False #to be True after it takes_off

# object detection screen dimension. note: (0,0) is the top-left screen corner
CENTER_SCREEN_X = 162.5
CENTER_SCREEN_Y = 122.5

#dictionary of gesture associated to a trajectory
recorded_gesture_dict = {}


####################### BASIC MODE gesture-action TRANSLATE ###########################
# Create a dictionary mapping BASIC gestures to actions
basic_gesture_to_action = {
    "Thumb_Up": "take_off",
    "Thumb_Down": "land",
    "ILoveYou": "start_trajectory",
    "Pointing_Up": "record_trajectory",
    "Victory": "victory",
    "Closed_Fist": "not_defined"
}
# Function to translate a gesture to its corresponding action
def basic_translate_gesture(gesture):
    return basic_gesture_to_action.get(gesture, "[BASIC_MODE: Unknown action")

####################### FUNCTION MODE gesture-action TRANSLATE ###########################
# Create a dictionary mapping FUNCTION gestures to actions
function_gesture_to_action = {
    "ILoveYou": "search_traj",
    "Pointing_Up": "take_photo",
    "Thumb_Up": "not_defined",
    "Thumb_Down": "not_defined",
    "Victory": "not_defined",
    "Closed_Fist": "not_defined"
}

# Function to translate a gesture to its corresponding action
def function_translate_gesture(gesture):
    return function_gesture_to_action.get(gesture, "[FUNCTION_MODE: Unknown action")



# Funzione di callback per la pressione dei tasti
def on_press(key):
    global running
    try:
        if key.char == 'w':
            print("Hai premuto 'w'. Il ciclo è terminato.")
            running = False
            return False  # Per interrompere il listener
    except AttributeError:
        pass
def no_motion(highlevel_commander, duration=0.4): #motion to be used when gesture not valid
    global flying
    if FLIGHT_MODE and flying:
        highlevel_commander.go_to(0.0, 0.0, 0.0, -math.pi/5, duration_s = duration, relative = True)
        time.sleep(0.4)
        highlevel_commander.go_to(0.0, 0.0, 0.0, 2*math.pi/5, duration_s = duration*2, relative = True)
        time.sleep(0.8)
        highlevel_commander.go_to(0.0, 0.0, 0.0, -math.pi/5, duration_s = duration, relative = True)
        time.sleep(0.4)
    else:
        print('not in FLIGHT_MODE, or it s not flying.')

def rotate_360(highlevel_commander, duration=2.0):
    global flying
    if FLIGHT_MODE and flying:
        highlevel_commander.go_to(0.0, 0.0, 0.0, math.pi, duration_s = duration/2, relative = True)
        time.sleep(duration)
        highlevel_commander.go_to(0.0, 0.0, 0.0, -math.pi, duration_s = duration/2, relative = True)
        time.sleep(duration)
    else:
        print('not in FLIGHT_MODE, or it s not flying.')

def victory_motion(hl_commander, duration=0.5):
    if FLIGHT_MODE and flying:
        hl_commander.go_to(0.0, 0.0, 0.2, 0.0, duration_s= duration, relative = True)
        time.sleep(duration)
        hl_commander.go_to(0.0, 0.0, -0.2, 0.0, duration_s= duration, relative = True)
        time.sleep(duration)
        hl_commander.go_to(0.0, 0.0, 0.2, 0.0, duration_s= duration, relative = True)
        time.sleep(duration)
        hl_commander.go_to(0.0, 0.0, -0.2, 0.0, duration_s= duration, relative = True)
        time.sleep(duration)
    else:
        print('not in FLIGHT_MODE, or it s not flying.')


def handle_basic_commands(gesture_command, hl_commander, cf, scf, result_queue):
    """Handles gestures in BASIC_COMMANDS mode."""
    global flying
    translated_command = basic_translate_gesture(gesture_command)
    if translated_command == 'take_off':
        print('[BASIC_COMMANDS] Takeoff command received')
        if FLIGHT_MODE:
            hl_commander.takeoff(0.8, 4.5)  # Adjust takeoff height and duration
        time.sleep(4.5)
        flying = True

    elif translated_command == 'land':
        print('[BASIC_COMMANDS] Land command received')
        if FLIGHT_MODE:    
            hl_commander.land(0.0, 4.5)  # Land smoothly
        time.sleep(4.5)
        flying = False

    elif translated_command == 'record_trajectory':
        print('[BASIC_COMMANDS] Record trajectory command received')
        print('Waiting for 1 parameter (gesture to associate to this new trajectory)...')
        try:
            #read new gesture to associate
            timeout_duration = 5
            start_time = time.time()
            new_traj_gesture = result_queue.get(timeout=timeout_duration)
            while (new_traj_gesture == 'Pointing_Up' or new_traj_gesture == 'None' or new_traj_gesture == 'ILoveYou'): #new gesture must be different than the command that takes it as parameter.
                elapsed_time = time.time() - start_time
                if elapsed_time > timeout_duration:
                    print("Gesture selection timed out!")
                    new_traj_gesture = None
                    break
                new_traj_gesture = result_queue.get(timeout=timeout_duration - elapsed_time)

            #record new trajectory
            if new_traj_gesture:
                dataTraj = []
                print(f'New trajectory will be associated with this gesture: {new_traj_gesture}')
                
                if FLIGHT_MODE and flying: 
                    hl_commander.land(0.0, 4.5)
                    flying = False
                time.sleep(6.0)
                recDuration_s = 15
                record_trajectory(cf, scf, recDuration_s)
                run_command('python3 ~/uav_trajectories/scripts/generate_trajectory.py my_timed_waypoints_yaw.csv traj.csv --pieces 5')
                print('New trajectory recorded')
                print('uploading trajectory...')
                header, dataTraj = import_csv('traj.csv')
                recorded_gesture_dict.update({new_traj_gesture : dataTraj})
                print(f'DICTIONARY: {recorded_gesture_dict}')
                cf.param.set_value("sound.effect", 7)
            else:
                print('No gesture received')
                no_motion(hl_commander, duration=0.4)
            
        except Empty:
            print("No gesture received in the given timeout period.")
            no_motion(hl_commander, duration=0.4)

        time.sleep(3.0)


    elif translated_command == 'start_trajectory':
        print('[BASIC_COMMANDS] Start trajectory command received')
        #run_sequence(cf, 1, 10)  # Run trajectory ID 1 for 10 seconds
        print('Waiting for 1 parameter (the trajectory you want to run)...')
        try:
            #read new gesture to associate
            timeout_duration = 5
            start_time = time.time()
            selected_traj_gesture = result_queue.get(timeout=timeout_duration)
            while (selected_traj_gesture == 'ILoveYou' or selected_traj_gesture == 'None'): #new gesture must be different than the command that takes it as parameter.
                elapsed_time = time.time() - start_time
                if elapsed_time > timeout_duration:
                    print("Trajectory selection timed out!")
                    selected_traj_gesture = None
                    break
                selected_traj_gesture = result_queue.get(timeout=timeout_duration - elapsed_time)
            print(f'gesture: {selected_traj_gesture} received. Looking if corresponding trajectory exists')

            if selected_traj_gesture in recorded_gesture_dict:
                print('associated trajectory exists. Uploading to cf...')
                trajectory_id = 1 #for now we overwrite the trajectory every time, so there s no more than 1 trajectory on the crazyflie.
                traj = recorded_gesture_dict[selected_traj_gesture]
                duration = upload_trajectory(cf, trajectory_id, traj) #duration=around 15s
                print('done! duration: ',duration)
                run_sequence(cf, trajectory_id=1, duration=duration)
                time.sleep(duration)
            else:
                print('Invalid gesture. No corresponding trajectory associated')
                no_motion(hl_commander, duration=0.4)

        except Empty:
            print("No gesture received in the given timeout period.")
            no_motion(hl_commander, duration=0.4)

    elif translated_command == 'victory':
        print('[BASIC_COMMANDS] Victory gesture received')
        # Add custom logic for victory gesture (✌️)
        param_name = "sound.effect"
        param_value = 4
        cf.param.set_value(param_name, param_value)
        #no_motion(hl_commander, duration=0.4)
        #time.sleep(3.0)
        victory_motion(hl_commander)
        time.sleep(3)
        #rotate_360(hl_commander)

def handle_function_level_commands(gesture_command, hl_commander, cf, result_gesture_queue, result_object_queue, obj_detection_event):
    """Handles gestures in FUNCTION_MODE mode."""
    translated_command = function_translate_gesture(gesture_command)
    if translated_command == 'search_traj':
        print('[FUNCTION MODE] Searching trajectory...')
        # Implement search trajectory logic
        print('Waiting for parameters...')
        params = []
        #parameter 1 - gesture. note: it must be in the dictionary of gesture-traj already recorded
        print('waiting for param1')
        time.sleep(1.0)
        param_name = "sound.effect"
        param_value = 3
        cf.param.set_value(param_name, param_value)
        time.sleep(1.0)
        cf.param.set_value(param_name, param_value)
        time.sleep(1.0)
        param_value = 6
        cf.param.set_value(param_name, param_value)
        last = None
        t = time.time()
        while not result_gesture_queue.empty():
            last = result_gesture_queue.get_nowait()
        print(time.time()-t)
        selected_traj = None
        if last != None and last != 'ILoveYou' and last != 'Pointing_Up' and last != 'None':
            if last in recorded_gesture_dict:
                print(f'Successfully selected {last} gesture. The corresponding trajectory exists.')
                selected_traj = recorded_gesture_dict[last]
                params.append(last)
            else:
                print('The Gesture set as param1 has no correponding trajectory recorded')
                no_motion(hl_commander)
                return
        else:
            print('Invalid gesture (it cannot be: None, ILoveYou, Pointing_Up)')
            return       
        
        time.sleep(1.0)

        #parameter 2. note: if you are here than the param1 is valid
        print('waiting for param2')
        param_value = 3
        cf.param.set_value(param_name, param_value)
        time.sleep(1.0)
        cf.param.set_value(param_name, param_value)
        time.sleep(1.0)
        param_value = 6
        cf.param.set_value(param_name, param_value)
        print(result_object_queue.qsize())
        obj_detection_event.set()
        print('show the object to the camera...you have 2 seconds')
        time.sleep(2.0) #the object must be show to the camera in 2 seconds.
        obj_detection_event.clear()
        print(result_object_queue.qsize())
        detected_object_list = []
        if result_object_queue.empty():
            print('no object detected.')
            no_motion(hl_commander)
            return
        else:
            while not result_object_queue.empty():
                detected_object_list.append(result_object_queue.get_nowait())

            print(f'number of object detected in the 2s interval : {result_object_queue.qsize()}')
            selected_object = None
            min_distance = float('inf') #max distance

            for detected_object in detected_object_list:
                # Calculate the center of the bounding box
                bbox_origin_x = detected_object.bounding_box.origin_x
                bbox_origin_y = detected_object.bounding_box.origin_y
                bbox_width = detected_object.bounding_box.width
                bbox_height = detected_object.bounding_box.height
                
                center_bbox_x = bbox_origin_x + bbox_width / 2
                center_bbox_y = bbox_origin_y + bbox_height / 2

                # calculate distance from the center of the screen
                distance = math.sqrt((center_bbox_x - CENTER_SCREEN_X) ** 2 + (center_bbox_y - CENTER_SCREEN_Y) ** 2)
                print(f'Object {detected_object.categories[0].category_name}: Distance from the center: {distance}')

                # Check if this object is the closer to the center respect to the others
                if distance < min_distance:
                    min_distance = distance
                    selected_object = detected_object

            if selected_object:
                print(f'Selected object is the one closest to the center: {selected_object.categories[0].category_name}')
                print(f'Bounding box: x={selected_object.bounding_box.origin_x}, y={selected_object.bounding_box.origin_y}, width={selected_object.bounding_box.width}, height={selected_object.bounding_box.height}')
                obj_to_find = selected_object.categories[0].category_name
                params.append(obj_to_find)
                print(f'Executing the action SearchTraj({params})')
                trajectory_id=1
                duration = upload_trajectory(cf, trajectory_id, selected_traj) #duration=around 15s
                print('uploading done! duration: ',duration)
                print('starting executing trajectory...')
                run_sequence(cf, trajectory_id=1, duration=duration)
                time.sleep(4.0)
                
                timeout_duration = duration
                start_time = time.time()
                obj_detection_event.set()
                obj_found = False
                while (time.time()-start_time) < timeout_duration and not obj_found:
                    if not result_object_queue.empty():
                        detection= result_object_queue.get(timeout_duration - (time.time()- start_time)) #TODO check again this parameters/ maybe also get_nowait works
                        detected_obj = detection.categories[0].category_name
                        print(f'found this object: {detected_obj}')
                        if detected_obj == obj_to_find:
                            obj_found = True
                            print(f'Object you were looking for: {obj_to_find} found. Interrupting the trajectory')
                            if FLIGHT_MODE:
                                hl_commander.go_to(0.0, 0.0, 0.8, 0.0, 4.0)
                            time.sleep(4.0)
                            victory_motion(hl_commander=hl_commander)
                            break
                if not obj_found:
                    if FLIGHT_MODE:
                        hl_commander.go_to(0.0, 0.0, 0.8, 0.0, 4.0)
                        time.sleep(4.0)
                    print(f'Obj {obj_to_find} not found in the whole trajectory')
                    no_motion(hl_commander)


            else: #it shouldn go inside here, because at this point at least 1 object must be selected.
                print('No object was selected. Shouldn t see this print')
                no_motion(hl_commander)
                return
        


    elif translated_command == 'take_photo':
        print('[FUNCTION MODE] take_photo command received')
        # Implement take_photo logic
        print('Waiting for parameters...')
        params = []
        #parameter 1 - gesture. note: it must be in the dictionary of gesture-traj already recorded
        print('waiting for param1')
        time.sleep(1.0)
        param_name = "sound.effect"
        param_value = 3
        cf.param.set_value(param_name, param_value)
        time.sleep(1.0)
        cf.param.set_value(param_name, param_value)
        time.sleep(1.0)
        param_value = 6
        cf.param.set_value(param_name, param_value)
        last = None
        t = time.time()
        while not result_gesture_queue.empty():
            last = result_gesture_queue.get_nowait()
        print(time.time()-t)
        selected_traj = None
        if last != None and last != 'ILoveYou' and last != 'Pointing_Up' and last != 'None':
            if last in recorded_gesture_dict:
                print(f'Successfully selected {last} gesture. The corresponding trajectory exists.')
                selected_traj = recorded_gesture_dict[last]
                params.append(last)
            else:
                print('The Gesture set as param1 has no correponding trajectory recorded')
                no_motion(hl_commander)
                return
        else:
            print('Invalid gesture (it cannot be: None, ILoveYou, Pointing_Up)')
            return       
        
        time.sleep(1.0)
        print(f'Executing the action SearchTraj({params})')
        trajectory_id=1
        duration = upload_trajectory(cf, trajectory_id, selected_traj) #duration=around 15s
        print('done! duration: ',duration)
        print('starting executing trajectory...')
        run_sequence(cf, trajectory_id=1, duration=duration)
        time.sleep(duration)


    elif translated_command == 'not_defined':
        print('[FUNCTION MODE] A command not defined yet')
        # Implement custom actions specific to FUNCTION_LEVEL
        #time.sleep(10.0)

        
def upload_trajectory(cf, trajectory_id, trajectory):
        trajectory_mem = cf.mem.get_mems(MemoryElement.TYPE_TRAJ)[0]
        trajectory_mem.trajectory = []

        total_duration = 0
        for row in trajectory:
            duration = row[0]
            x = Poly4D.Poly(row[1:9])
            y = Poly4D.Poly(row[9:17])
            z = Poly4D.Poly(row[17:25])
            yaw = Poly4D.Poly(row[25:33])
            trajectory_mem.trajectory.append(Poly4D(duration, x, y, z, yaw))
            total_duration += duration

        upload_result = trajectory_mem.write_data_sync()
        if not upload_result:
            print('Upload failed, aborting!')
            sys.exit(1)
        cf.high_level_commander.define_trajectory(trajectory_id, 0, len(trajectory_mem.trajectory))
        return total_duration

def record_trajectory(cf, scf, duration_s):
    lg_stab = LogConfig(name='Stabilizer', period_in_ms=500)
    lg_stab.add_variable('stateEstimate.x', 'float')
    lg_stab.add_variable('stateEstimate.y', 'float')
    lg_stab.add_variable('stateEstimate.z', 'float')
    lg_stab.add_variable('stateEstimate.yaw', 'float')

    timed_waypoints = {'t': [], 'x': [], 'y': [], 'z': [], 'yaw': []}

    param_name = "sound.effect"
    param_value = 3
    cf.param.set_value(param_name, param_value)
    print('3')
    time.sleep(1)
    print('2')
    cf.param.set_value(param_name, param_value)
    time.sleep(1)
    print('1')
    cf.param.set_value(param_name, param_value)
    time.sleep(1)
    print('start recording...')
    cf.param.set_value(param_name, 6)
    time.sleep(1)

    with SyncLogger(scf, lg_stab) as logger:
        startTime = time.time()
        endTime = startTime + duration_s

        for log_entry in logger:
            timestamp = log_entry[0]
            data = log_entry[1]
            logconf_name = log_entry[2]

            timed_waypoints['t'].append(time.time()-startTime)
            timed_waypoints['x'].append(data['stateEstimate.x'])
            timed_waypoints['y'].append(data['stateEstimate.y'])
            timed_waypoints['z'].append(data['stateEstimate.z'])
            timed_waypoints['yaw'].append(data['stateEstimate.yaw'])

            print('[%d][%s]: %s' % (timestamp, logconf_name, data))

            if time.time() > endTime:
                # Plotting the trajectory
                # fig = plt.figure()
                # ax = fig.add_subplot(111, projection='3d')
                # ax.plot(timed_waypoints['x'],
                #         timed_waypoints['y'], timed_waypoints['z'])
                # ax.set_xlabel('X')
                # ax.set_ylabel('Y')
                # ax.set_zlabel('Z')
                # ax.set_title('Crazyflie Trajectory')
                # plt.show()

                cf.param.set_value(param_name, 6)
                save_to_csv(timed_waypoints, 'my_timed_waypoints_yaw.csv')
                print('csv Saved')
                break

def save_to_csv(timed_waypoints, filename):
    """Save the positions to a CSV file"""
    with open(filename, 'w', newline='') as csvfile:
        fieldnames = ['t', 'x', 'y', 'z', 'yaw']  # ['x', 'y', 'z']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for i in range(len(timed_waypoints['x'])):
            writer.writerow({'t': timed_waypoints['t'][i],
                            'x': timed_waypoints['x'][i],
                            'y': timed_waypoints['y'][i],
                            'z': timed_waypoints['z'][i],
                            'yaw': timed_waypoints['yaw'][i],  #without yaw put: 0
                            })

def import_csv(filename):
    """
    Import data from a CSV file and return it as a list of lists.
    Assumes the first row is a header.
    """
    data = []
    with open(filename, 'r', newline='') as csvfile:
        reader = csv.reader(csvfile)
        header = next(reader)  # Read the header
        for row in reader:
            data.append([float(value) for value in row])
    return header, data
def run_command(command):
    #command = 'python3 ~/uav_trajectories/scripts/generate_trajectory.py my_timed_waypoints_yaw.csv traj.csv --pieces 5'
    
    # Stampa il comando che verrà eseguito
    print("Running command:", command)
    
    try:
        # Esegui il comando e attendi che termini
        result = subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, shell=True)
        
        # Stampa l'output del comando
        print("Command output:")
        print(result.stdout)
        
        # Se ci sono errori, stampa gli errori
        if result.stderr:
            print("Command errors:")
            print(result.stderr)
    
    except subprocess.CalledProcessError as e:
        # In caso di errore, stampa il codice di ritorno e l'output
        print(f"Command failed with return code {e.returncode}")
        print("Error output:")
        print(e.stderr)
        print("Full command output:")
        print(e.output)

def run_sequence(cf, trajectory_id, duration):
    global flying
    if FLIGHT_MODE:
        commander = cf.high_level_commander
        if not flying:
            commander.takeoff(0.4, 2.0)
            time.sleep(2.0)
            flying=True
        #commander.go_to(0.0, 0.0, 0.6, 0.0, 6.0)
        #time.sleep(3.0)
        relative = True
        commander.start_trajectory(trajectory_id, time_scale=1.0, relative=relative)
        #time.sleep(duration) #NOTE: i put this outside, so i can read the queue of the object in the meanwhile, and i m able to interrupt the trajectory with a goto if an object is detected
        #commander.land(0.0, 2.0)
        #commander.stop()

def wait_for_position_estimator(scf):
    print('Waiting for estimator to find position...')

    log_config = LogConfig(name='Kalman Variance', period_in_ms=500)
    log_config.add_variable('kalman.varPX', 'float')
    log_config.add_variable('kalman.varPY', 'float')
    log_config.add_variable('kalman.varPZ', 'float')

    var_y_history = [1000] * 10
    var_x_history = [1000] * 10
    var_z_history = [1000] * 10

    threshold = 0.001

    with SyncLogger(scf, log_config) as logger:
        for log_entry in logger:
            data = log_entry[1]

            var_x_history.append(data['kalman.varPX'])
            var_x_history.pop(0)
            var_y_history.append(data['kalman.varPY'])
            var_y_history.pop(0)
            var_z_history.append(data['kalman.varPZ'])
            var_z_history.pop(0)

            min_x = min(var_x_history)
            max_x = max(var_x_history)
            min_y = min(var_y_history)
            max_y = max(var_y_history)
            min_z = min(var_z_history)
            max_z = max(var_z_history)

            # print("{} {} {}".
            #       format(max_x - min_x, max_y - min_y, max_z - min_z))

            if (max_x - min_x) < threshold and (
                    max_y - min_y) < threshold and (
                    max_z - min_z) < threshold:
                break

def reset_estimator(cf):
    cf.param.set_value('kalman.resetEstimation', '1')
    time.sleep(0.1)
    cf.param.set_value('kalman.resetEstimation', '0')

    wait_for_position_estimator(cf)


def command_crazyflie(result_gesture_queue, result_object_queue, obj_detection_event): #result_queue,event
    print('starting command cf')
    state = 'BASIC_COMMANDS'  # Initial state. then we have FUNCTION_LEVEL, and BUSY (when i m doing something)
    recent_gestures = []  # List to keep track of recent gestures
    last_gesture = ''

    


    # Ottieni il PID del processo corrente
    current_pid = os.getpid()
    print("PID del processo loopProva:", current_pid)


    # time.sleep(5)

    cflib.crtp.init_drivers()

    # header, dataTraj = import_csv('traj1.csv')
    # print(dataTraj)  

    # Avvia il listener per i tasti
    listener = keyboard.Listener(on_press=on_press)
    listener.start()


    # URI to the Crazyflie to connect to
    uri = uri_helper.uri_from_env(default='radio://0/125/2M/E7E7E7E7E7')
    #uri = uri_helper.uri_from_env(default='usb://0')

    print(' uri setted')


    


    with SyncCrazyflie(uri, cf=Crazyflie(rw_cache='./cache')) as scf:
        cf = scf.cf
        if FLIGHT_MODE:
            reset_estimator(cf)
        hl_commander = cf.high_level_commander

        i = 0

        print('cfProcess: starting loop..')
        while running:

            if not result_gesture_queue.empty():
                gesture_command = result_gesture_queue.get()
                if(gesture_command == last_gesture):
                    continue
                last_gesture = gesture_command
                print(f"Received gesture: {gesture_command}")

                # Add recognized gesture to the recent_gestures list
                if gesture_command != 'None': #None sometimes comes out when we change the gesture, so we dont take it into account
                    recent_gestures.append(gesture_command)
                if len(recent_gestures) > 4 and gesture_command:  # Keep only the last 4 gestures
                    recent_gestures.pop(0)

                # Check if the palm-fist-palm-fist sequence is detected
                if recent_gestures == ['Open_Palm', 'Closed_Fist', 'Open_Palm', 'Closed_Fist']:
                    if state == 'BASIC_COMMANDS':
                        print('Switching to FUNCTION_LEVEL mode')
                        state = 'FUNCTION_LEVEL'
                        cf.param.set_value("sound.effect", 12)
                        time.sleep(1)
                        cf.param.set_value("sound.effect", 0)

                    else:
                        print('Switching to BASIC_COMMANDS mode')
                        state = 'BASIC_COMMANDS'
                        cf.param.set_value("sound.effect", 12)
                        time.sleep(1)
                        cf.param.set_value("sound.effect", 0)

                    recent_gestures.clear()  # Reset sequence after transition

                if gesture_command == 'Open_Palm' or gesture_command == 'None' or gesture_command == 'Closed_Fist':
                    continue
                # Handle commands based on the current modecommandercf
                if state == 'BASIC_COMMANDS':
                    state = 'BUSY'
                    handle_basic_commands(gesture_command, hl_commander, cf, scf, result_gesture_queue)
                    #print(f'queue size (gesture recognized during an action: {result_gesture_queue.qsize()}')
                    while not result_gesture_queue.empty(): #try to empty the queue (because during the action it may have recognize some gestures. it should be faster than the feeder, so it shouldn t block.
                        result_gesture_queue.get_nowait()
                    #print('Trying to empty the queue...')
                    #print(f'queue size (should be 0): {result_gesture_queue.qsize()}')
                    state = 'BASIC_COMMANDS'
                elif state == 'FUNCTION_LEVEL':
                    state = 'BUSY' 
                    handle_function_level_commands(gesture_command, hl_commander, cf, result_gesture_queue, result_object_queue, obj_detection_event)
                    while not result_gesture_queue.empty():
                        result_gesture_queue.get_nowait() 
                    state = 'FUNCTION_LEVEL'

            #note: BUSY state is not actually used.

                # print(result_object_queue.qsize())
                # obj_detection_event.set()
                # time.sleep(3)
                # print(result_object_queue.qsize())
                # obj_detection_event.clear()
                # while not result_object_queue.empty():
                #     detect_object = result_object_queue.get_nowait()
                #     print(detect_object.__dict__)
