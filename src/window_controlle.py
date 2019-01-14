
import pyautogui
from win32 import win32gui
import cv2

### parameters ###

# active_window
game_window_pos_x = 1100
game_window_pos_y = 40

# adjust_window_pos_size
win_left = 1000
win_top = 0
win_width = 200
win_height = 180

### functions ###
def active_window():
    pyautogui.click(game_window_pos_x, game_window_pos_y)

def adjust_window_pos_size(win_left, win_top, win_width, win_height):
    hwnd = win32gui.FindWindow("wxWindowNR",None)
    win32gui.MoveWindow(hwnd, win_left, win_top, win_width, win_height, True)

def imshow(frame):
    cv2.imshow('window',frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

