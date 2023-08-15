import cv2
import mediapipe as mp
import math

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_holistic = mp.solutions.holistic

FINGERTIP_INDEX = 8  # 食指指尖的索引
THUMB_INDEX = 4  # 大拇指指尖的索引

# 记录圆心坐标和半径
circle_center = None
circle_radius = None

cap = cv2.VideoCapture(0)
with mp_holistic.Holistic(
        min_detection_confidence=0.4,
        min_tracking_confidence=0.4) as holistic:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = holistic.process(image)

        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

        # 获取右手食指和大拇指的坐标
        if results.right_hand_landmarks:
            hand_landmarks = results.right_hand_landmarks.landmark
            thumb_tip = hand_landmarks[THUMB_INDEX]
            index_fingertip = hand_landmarks[FINGERTIP_INDEX]

            # 计算食指和大拇指之间的距离
            distance = math.sqrt(
                (thumb_tip.x - index_fingertip.x) ** 2 + (thumb_tip.y - index_fingertip.y) ** 2
            )

            if distance > 0.1:
                # 更新圆心坐标和半径
                circle_center = (int(index_fingertip.x * image.shape[1]), int(index_fingertip.y * image.shape[0]))
                circle_radius = int(distance * image.shape[1] / 2)
                cv2.circle(image, circle_center, circle_radius, (0, 255, 0), 2)
                cv2.waitKey(100)

        cv2.imshow('MediaPipe Holistic', cv2.flip(image, 1))
        if cv2.waitKey(5) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()