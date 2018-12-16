
import numpy as np
import cv2
from grabscreen import grab_screen
#from directkeys import *
import pyautogui
from directkeys import *
from window_controlle import active_window

A = 0x1E
B = 0x30
leftarrow = 0xcb
rightarrow = 0xcd
uparrow = 0xc8
downarrow = 0xd0

keys_to_press = [A, B, leftarrow, rightarrow, uparrow, downarrow]
keys_name = ['A', 'B', 'leftarrow', 'rightarrow', 'uparrow', 'downarrow']

num_actions = len(keys_to_press)

# active game window
active_window()
#PressKey(A)


while True:

    # 画面取得
    #grab_screen()

    action = int(np.random.randint(0, num_actions, size=1))
    print('action: ', keys_name[action])

    PressKey(keys_to_press[action])
    time.sleep(0.5)
    ReleaseKey(keys_to_press[action])

    # q押したら終了
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

#画面消す
cv2.destroyAllWindows()

