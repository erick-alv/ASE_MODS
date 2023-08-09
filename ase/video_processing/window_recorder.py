#import pyautogui
import mss
import time
import cv2
import subprocess
import numpy as np


MAX_SEARCH_TRIES = 800

def search_for_window(window_name):
    window_id_str = None
    window_id_bytes = None
    num_tries = 0
    while window_id_str is None and num_tries < MAX_SEARCH_TRIES:
        try:
            ans = subprocess.check_output(['xdotool', 'search', '--onlyvisible', '--name', window_name])
            window_id_bytes = ans.strip()
            window_id_str = window_id_bytes.decode()
        except subprocess.CalledProcessError:
            pass

        time.sleep(0.01)
        num_tries += 1

    if window_id_str is None:
        print("Window not found.")
        print(num_tries)
        exit()
    else:
        print("Window found!!!!")
        return window_id_str, window_id_bytes

def check_window_open(window_name):#
    for i in range(5):
        try:
            #we just want that it trows the error when the window has been closed
            ans = subprocess.check_output(['xdotool', 'search', '--onlyvisible', '--name', window_name])
            break # if found before range ends will stop immediately
        except:
            if i == 4:
                raise


#we not use tra here since we want the exception to show when the window was closed
def get_window_geometry(window_id_str):
    ans = subprocess.check_output(['xdotool', 'getwindowgeometry', '--shell', window_id_str])
    window_info = dict(line.strip().split('=') for line in ans.decode().split('\n') if line.strip())
    return int(window_info['X']), int(window_info['Y']), int(window_info['WIDTH']), int(
        window_info['HEIGHT'])

def press_tab_on_window(window_id_str):
    #subprocess.call()
    command = f"xdotool windowactivate {window_id_str} && xdotool key Tab"
    subprocess.call(command, shell=True)
    print("Here")


def main():
    # Set the resolution and output file name
    screen_size = (1280, 720)
    # Create a video writer
    output_file = '/home/erick/Videos/exp_vids/window_recording.avi'
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    out = cv2.VideoWriter(output_file, fourcc=fourcc, fps=30.0, frameSize=screen_size)


    window_id_str, window_id_bytes = search_for_window('Isaac Gym')
    print(window_id_str)
    x, y, width, height = get_window_geometry(window_id_str)
    x -= 10
    y -= 45
    press_tab_on_window(window_id_str)



    #for tracking fps
    # prev = 0
    desired_fps = 30.0
    desired_dt = 1 / desired_fps

    start = None
    while True:
        try:
            with mss.mss() as sct:
                check_window_open('Isaac Gym')

                region = {'top': y, 'left': x, 'width': width, 'height': height}
                screenshot = sct.grab(region)

                frame = cv2.cvtColor(np.array(screenshot), cv2.COLOR_BGR2RGB)
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)#for some reason the first time does not work correctly
                # Write the captured frame to the video file


                if cv2.waitKey(1) == ord('q'):
                    break

                end = time.time()
                if start is not None:
                    dt = end - start
                    print("dt: ", dt, "; fps: ", 1.0/dt)
                    dt_diff = desired_dt - dt
                    dt_diff *= 0.8
                    if dt_diff > 0:
                        time.sleep(dt_diff)
                    out.write(frame)
                    #new_end = time.time()
                    #new_dt = new_end - start
                    #print("new dt: ", new_dt, "; new fps: ", 1.0 / new_dt)
                else:
                    out.write(frame)

                start = time.time()

        except subprocess.CalledProcessError:
             # Window closed or not visible anymore
            print("Window closed.")
            break

    # Release the video writer and close the window
    out.release()
    cv2.destroyAllWindows()





if __name__ ==  "__main__":
    main()