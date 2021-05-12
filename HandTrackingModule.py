import cv2
import mediapipe as mp
from cmath import phase


class HandDetector:
    def __init__(self, mode=False, maxHands=2, detectionCon=0.5, trackCon=0.5) -> None:
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.img = None
        self.landmarks = {}
        self.bbox = {}
        self.polarLandmarks = {}
        self.nHands = 0

        self.mpHands = mp.solutions.hands
        self.mpDraw = mp.solutions.drawing_utils

        self.hands = self.mpHands.Hands(
            self.mode, self.maxHands, self.detectionCon, self.trackCon
        )
        self.fingerIDs = [
            [4, 3, 2],
            [8, 7, 6, 5],
            [12, 11, 10, 9],
            [16, 15, 14, 13],
            [20, 19, 18, 17],
        ]

    def findHands(self, img, draw=True):
        self.img = img
        self.landmarks = {}
        self.polarLandmarks = {}
        self.bbox = {}
        self.nHands = 0
        # convert the image color system to RGB to be processed by the mediapipe library
        imgRGB = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)

        # Process the image in real time to find the hands (max is 2 by defaults)
        self.results = self.hands.process(imgRGB)

        # Draw lines per hand (max is 2 by defaults)
        if self.results.multi_hand_landmarks:
            self.nHands = len(self.results.multi_hand_landmarks)
            for handNo, handLMS in enumerate(self.results.multi_hand_landmarks):
                self.landmarks[handNo] = []
                self.polarLandmarks[handNo] = []
                self.bbox[handNo] = []
                if draw:
                    self.mpDraw.draw_landmarks(
                        self.img, handLMS, self.mpHands.HAND_CONNECTIONS
                    )

        return self.nHands

    def getLandmarks(self, handNo=0, showID=False, color=(255, 0, 255)):

        if self.nHands > handNo:
            myHand = self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(myHand.landmark):
                imgHight, imgWidth, _ = self.img.shape
                cx, cy = int(lm.x * imgWidth), int(lm.y * imgHight)
                self.landmarks[handNo].append((cx, cy))
                if showID:
                    cv2.putText(
                        self.img,
                        str(id),
                        (cx, cy),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        color,
                        2,
                    )
            return self.landmarks[handNo]
        return None

    def getFingersLandmarks(self, handNo=0):
        if self.nHands > handNo:
            if not self.landmarks[handNo]:
                self.getlandmarks(handNo)

            fingersLM = {
                finger: [self.landmarks[handNo][i] for i in f]
                for f, finger in zip(
                    self.fingerIDs, ["thumb", "index", "middle", "ring", "pinky"]
                )
            }
            return fingersLM
        return None

    def getPolarLandmarks(self, handNo=0):
        if self.nHands > handNo:
            if not self.landmarks[handNo]:
                self.getLandmarks(self.img.copy(), handNo)

            anchor = self.landmarks[handNo][0]
            self.polarLandmarks[handNo] = [
                (
                    phase(complex(p[0] - anchor[0], p[1] - anchor[1])),
                    abs(complex(p[0] - anchor[0], p[1] - anchor[1])),
                )
                for p in self.landmarks[handNo]
            ]
            return self.polarLandmarks[handNo]
        return None

    def getPolerFingersLandmarks(self, handNo=0):
        if self.nHands > handNo:
            if not self.polarLandmarks[handNo]:
                self.getPolarLandmarks( handNo)

            fingersLM = {
                finger: [self.polarLandmarks[handNo][i] for i in f]
                for f, finger in zip(
                    self.fingerIDs, ["thumb", "index", "middle", "ring", "pinky"]
                )
            }
            return fingersLM
        return None

    def getBoarderBox(self, handNo=0, draw=False, color=(0, 255, 0)):
        if self.nHands > handNo:
            if not self.landmarks[handNo]:
                self.getLandmarks(self.img.copy(), handNo)

            xList = [i[0] for i in self.landmarks[handNo]]
            yList = [i[1] for i in self.landmarks[handNo]]
            xmin, xmax = min(xList), max(xList)
            ymin, ymax = min(yList), max(yList)
            self.bbox[handNo] = xmin, ymin, xmax, ymax

            if draw:
                cv2.rectangle(
                    self.img,
                    (xmin - 20, ymin - 20),
                    (xmax + 20, ymax + 20),
                    color,
                    2,
                )
            return self.bbox[handNo]
        return None


##################################################


def main():
    # set the capture stream input
    #   1- path to file
    # or
    #   2- number indecatin the capture device
    #       0 => main device
    #       4 => loopback used with obs and DroidCam

    cap = cv2.VideoCapture(0)
    detector = HandDetector()

    while cap.isOpened():
        # capture the image stream fro source (0 is the main camera)
        success, img = cap.read()
        img = cv2.flip(img, 1)
        detector.findHands(img)

        detector.getLandmarks()
        detector.getBoarderBox(draw=True)
        fingers = detector.getFingersLandmarks()
        if fingers:
            cv2.circle(img, fingers["index"][0], 10, (255, 0, 255), 3)

        # display the results live on realtime
        cv2.imshow("hand recognition", img)

        # Time in ms to wait before frames update (0 => do not update)
        cv2.waitKey(1)
        if cv2.getWindowProperty("hand recognition", 4) < 1:
            cap.release()


if __name__ == "__main__":
    main()
