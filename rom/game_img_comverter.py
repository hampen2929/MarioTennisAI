
import cv2
from grabscreen import grab_screen

while True:
    # 画面取得

    grab_screen()

    # q押したら終了
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

#画面消す
cv2.destroyAllWindows()

