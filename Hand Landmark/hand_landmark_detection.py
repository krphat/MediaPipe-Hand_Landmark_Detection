
import cv2
import mediapipe as mp
import numpy as np

class HandLandmarker():
    def __init__(self, mode = False, maxHands=1, minDetectionCon=0.5, minTrackCon=0.5):
        self.__mode = mode
        self.__maxHands = maxHands
        self.__minDetectionCon = minDetectionCon
        self.__minTrackCon = minTrackCon

        self.__mpHands = mp.solutions.hands
        self.__hands = self.__mpHands.Hands(self.__mode, self.__maxHands, self.__minDetectionCon, self.__minTrackCon)
        self.__mpDraw = mp.solutions.drawing_utils

    def detectHand(self, image):
        hands_appear = []
        image = cv2.flip(image, 1)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.__result = self.__hands.process(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if self.__result.multi_hand_landmarks:
            for id, hands_lms in enumerate(self.__result.multi_hand_landmarks):
                label = self.__result.multi_handedness[id].classification[0].label
                if label not in hands_appear:
                    hands_appear.append(label)
                coord = tuple(np.multiply(
                        np.array((hands_lms.landmark[self.__mpHands.HandLandmark.WRIST].x, hands_lms.landmark[self.__mpHands.HandLandmark.WRIST].y)),
                        [640, 480]).astype(int))
                cv2.putText(image, label, coord, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                self.__mpDraw.draw_landmarks(image, hands_lms, self.__mpHands.HAND_CONNECTIONS)
        hands_appear.sort()
        return image, hands_appear
    
    def findPosition(self, image):
        lstPositon = []
        if self.__result.multi_hand_landmarks:
            for hands_lms in self.__result.multi_hand_landmarks:
                for kp_idx, lm in enumerate(hands_lms.landmark):
                    h, w, c = image.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    lstPositon.append([kp_idx, cx, cy])
        return lstPositon
    
    def getLandmarks(self):
        listLandmarks = []
        if self.__result.multi_hand_landmarks:
            for hands_lms in self.__result.multi_hand_landmarks:
                for kp_idx, lm in enumerate(hands_lms.landmark):
                    listLandmarks.append([kp_idx, lm.x, lm.y])
        return listLandmarks