import mediapipe as mp
from cmath import phase


class HandDetector:
    def __init__(
        self, STATIC_IMAGE_MODE=False, maxHands=2, detectionCon=0.6, trackCon=0.5
    ) -> None:
        """Initalize the hand detector

                                        Args             :
                      STATIC_IMAGE_MODE (bool, optional) : [description]. Defaults to False.
                      maxHands          (int, optional)  : Maximum number of hands to detect. Defaults to 2.
                      detectionCon      (float, optional): Minimum confidence value ([0.0, 1.0]) from the hand detection model
                                            for the detection to be considered successful. Defaults to 0.6.
        trackCon (float, optional): Minimum confidence value ([0.0, 1.0]) from the landmark-tracking model
                                            for the hand landmarks to be considered tracked successfully, or otherwise hand detection will be invoked automatically on the next input image. Defaults to 0.5.
        """
        self.img = None
        self.landmarks = None
        self.bbox = None
        self.polarLandmarks = None
        self.nHands = 0
        self.fingerIDs = [
            [4, 3, 2],
            [8, 7, 6, 5],
            [12, 11, 10, 9],
            [16, 15, 14, 13],
            [20, 19, 18, 17],
        ]
        self.mpHands = mp.solutions.hands
        self.mpDraw = mp.solutions.drawing_utils

        self.hands = self.mpHands.Hands(
            STATIC_IMAGE_MODE, maxHands, detectionCon, trackCon
        )

    def find_hands(self, img, drawHandConnections=True):
        """Process input image to find, detect and, locate hands

                            Args                  :
        img                 (numpy-like 3d tensor): image to be processed
        drawHandConnections (bool, optional)      : (if true) draw the connection between landmarks on the image
                                                    for debuging and visualization. Defaults to True.

        Returns:
        int    : number of hands found in the image
        """
        self.img = img  # store the img in a local attribute
        self.nHands = 0  # set the number of hands to 0 as defualt

        # Convert the image color system to RGB then Process real time to detect and locate the hands (max is 2 by defaults)
        self.results = self.hands.process(cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB))

        if self.results.multi_hand_landmarks:
            self.nHands = len(self.results.multi_hand_landmarks)
            # reset the landmarks to an empty lists
            self.landmarks = [None] * self.nHands
            # reset the polarLandmarks to an empty lists
            self.polarLandmarks = [None] * self.nHands
            # reset the bbox to an empty lists
            self.bbox = [None] * self.nHands

            # Draw lines for each hand
            if drawHandConnections:
                for handLMS in self.results.multi_hand_landmarks:
                    self.mpDraw.draw_landmarks(
                        self.img, handLMS, self.mpHands.HAND_CONNECTIONS
                    )

        return self.nHands

    def get_landmarks(self, handID=0):
        """find and return the coordinates of all the landmarks for spcific and

               Args           :
        handID (int, optional): hand ID (0..n). Defaults to 0.

             Returns:
        list or None: list of coordinates for all the landmarks or None if no hand detected
        """
        if self.nHands > handID:
            myHand = self.results.multi_hand_landmarks[handID]
            imgHight, imgWidth, _ = self.img.shape

            self.landmarks[handID] = [
                (int(lm.x * imgWidth), int(lm.y * imgHight)) for lm in myHand.landmark
            ]
            return self.landmarks[handID]
        return None

    def get_fingers_landmarks(self, handID=0):
        """find the landmarks for each finger

               Args           :
        handID (int, optional): hand ID (0..n). Defaults to 0.

             Returns:
        dict or None: dictionary of list of landmarks for each finger ("thumb", "index", "middle", "ring", "pinky")
        """
        if self.nHands > handID:
            if not self.landmarks[handID]:
                self.get_landmarks(handID)

            return {
                finger: [self.landmarks[handID][i] for i in f]
                for f, finger in zip(
                    self.fingerIDs, ["thumb", "index", "middle", "ring", "pinky"]
                )
            }
        return None

    def get_polar_landmarks(self, handID=0):
        """get the polar coordinates (angle, distance) for the wrist point

               Args           :
        handID (int, optional): [description]. Defaults to 0.

        Returns:
        list   : list of polar coordinates (angle in radian)
        """
        if self.nHands > handID:
            if not self.landmarks[handID]:
                self.get_landmarks(self.img.copy(), handID)

            anchor = self.landmarks[handID][0]
            self.polarLandmarks[handID] = [
                (
                    phase(complex(p[0] - anchor[0], p[1] - anchor[1])),
                    abs(complex(p[0] - anchor[0], p[1] - anchor[1])),
                )
                for p in self.landmarks[handID]
            ]
            return self.polarLandmarks[handID]
        return None

    def get_polar_fingers_landmarks(self, handID=0):
        """get the Polar coordinates for each finger

               Args           :
        handID (int, optional): [description]. Defaults to 0.

             Returns:
        dict or None: dictinary of list of landmarks polar coordinates (angle, distance) for each finger ("thumb", "index", "middle", "ring", "pinky")
        """
        if self.nHands > handID:
            if not self.polarLandmarks[handID]:
                self.get_polar_landmarks(handID)

            return {
                finger: [self.polarLandmarks[handID][i] for i in f]
                for f, finger in zip(
                    self.fingerIDs, ["thumb", "index", "middle", "ring", "pinky"]
                )
            }
        return None

    def get_bounding_box(self, handID=0, drawBox=False, color=(255, 255, 0)):
        """Calculate and return the bounding box for the given hand

                Args                  :
        img     (numpy-like 3d tensor): image to be processed
        drawBox (bool, optional)      : (if true) draw the connection between landmarks on the image
                                                    for debuging and visualization. Defaults to True.
            color (tuple, optional): bounding box color. Defaults to green(0, 255, 0). [works only if drawBox is set to True]

              Returns:
        tuple or None: (x, y, hight, width) of the bounding box
        """
        if self.nHands > handID:
            if not self.landmarks[handID]:
                self.get_landmarks(self.img.copy(), handID)

            xList = [i[0] for i in self.landmarks[handID]]
            yList = [i[1] for i in self.landmarks[handID]]
            x, xmax = min(xList), max(xList)
            y, ymax = min(yList), max(yList)
            w, h = xmax - x, ymax - y
            self.bbox[handID] = x, y, w, h

            if drawBox:
                cv2.rectangle(
                    self.img,
                    (x, y),
                    (x + w, y + h),
                    color,
                    1,
                )
            return self.bbox[handID]
        return None


##################################################

### Demo of HandDetector
import cv2


def main():
    # set the capture stream input
    #   - path to file
    # or
    #   - number indecatin the capture device
    # 0 => main device
    # 4 => loopback used with obs and DroidCam

    cap = cv2.VideoCapture(0)
    detector = HandDetector()

    while cap.isOpened():
        # capture the image stream fro source (0 is the main camera)
        success, img = cap.read()
        img = cv2.flip(img, 1)
        detector.find_hands(img)

        detector.get_landmarks()
        detector.get_bounding_box(drawBox=True)
        fingers = detector.get_fingers_landmarks()
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
