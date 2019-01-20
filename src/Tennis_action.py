# direct inputs
# source to this solution and code:
# http://stackoverflow.com/questions/14489013/simulate-python-keypresses-for-controlling-a-game
# http://www.gamespp.com/directx/directInputKeyboardScanCodes.html

import ctypes
import time
import numpy as np
import itertools

SendInput = ctypes.windll.user32.SendInput
# keys memory address
#http://geck.bethsoft.com/index.php?title=Template:DirectX_Scancodes

A = 0x1E # top spin shot
B = 0x30 # slice spin shot

leftarrow = 0xcb # move to left
rightarrow = 0xcd # move to right
uparrow = 0xc8 # move to upper
downarrow = 0xd0 # move to lower
stay = 0x0B # 0: stay
noshot = 0x0A # 9: noshot

F = 0x21 # advanced frame

# keys list

keys_move_to_press = [leftarrow, rightarrow, stay]
keys_move_name = ['left', 'right', 'stay']

keys_shot_to_press = [A, noshot]
keys_shot_name = ['A', 'noshot']

#sleep_time = [0.3]

num_move_actions = len(keys_move_to_press)
num_shot_actions = len(keys_shot_to_press)
#num_sleep_time = len(sleep_time)

keys_to_press = list(itertools.product(keys_move_to_press, keys_shot_to_press))
keys_to_press_name = list(itertools.product(keys_move_name, keys_shot_name))

num_actions = len(keys_to_press)

# C struct redefinitions
PUL = ctypes.POINTER(ctypes.c_ulong)


class KeyBdInput(ctypes.Structure):
    _fields_ = [("wVk", ctypes.c_ushort),
                ("wScan", ctypes.c_ushort),
                ("dwFlags", ctypes.c_ulong),
                ("time", ctypes.c_ulong),
                ("dwExtraInfo", PUL)]


class HardwareInput(ctypes.Structure):
    _fields_ = [("uMsg", ctypes.c_ulong),
                ("wParamL", ctypes.c_short),
                ("wParamH", ctypes.c_ushort)]


class MouseInput(ctypes.Structure):
    _fields_ = [("dx", ctypes.c_long),
                ("dy", ctypes.c_long),
                ("mouseData", ctypes.c_ulong),
                ("dwFlags", ctypes.c_ulong),
                ("time", ctypes.c_ulong),
                ("dwExtraInfo", PUL)]


class Input_I(ctypes.Union):
    _fields_ = [("ki", KeyBdInput),
                ("mi", MouseInput),
                ("hi", HardwareInput)]


class Input(ctypes.Structure):
    _fields_ = [("type", ctypes.c_ulong),
                ("ii", Input_I)]


# Actuals Functions

def PressKey(hexKeyCode):
    extra = ctypes.c_ulong(0)
    ii_ = Input_I()
    ii_.ki = KeyBdInput(0, hexKeyCode, 0x0008, 0, ctypes.pointer(extra))
    x = Input(ctypes.c_ulong(1), ii_)
    ctypes.windll.user32.SendInput(1, ctypes.pointer(x), ctypes.sizeof(x))


def ReleaseKey(hexKeyCode):
    extra = ctypes.c_ulong(0)
    ii_ = Input_I()
    ii_.ki = KeyBdInput(0, hexKeyCode, 0x0008 | 0x0002, 0, ctypes.pointer(extra))
    x = Input(ctypes.c_ulong(1), ii_)
    ctypes.windll.user32.SendInput(1, ctypes.pointer(x), ctypes.sizeof(x))

def action_random():
    # 行動決定
    action = int(np.random.randint(0, num_actions, size=1))
    return action

def take_action(action):
    actions = keys_to_press[action]

    #print(keys_to_press_name[action])

    # ボタン選択
    PressKey(actions[0])  # move
    time.sleep(0.3)
    ReleaseKey(actions[0])  # move

    PressKey(actions[1])  # shot
    time.sleep(0.2)
    ReleaseKey(actions[1])  # shot


