#!/usr/bin/env python3

import HandTrackingModule as htm
import cv2


def main():
    cap = cv2.VideoCapture(0)  # Capture webcam video stream
    detector = htm.HandDetector(maxHands=2)  # Initialize HandDetector

    while cap.isOpened():  # Loop while webcam is open

        _, img = cap.read()  # read images from the webcam
        # mirro the image horizontally (removing this line will not break the program)
        img = cv2.flip(img, 1)
        # process the image to find, detect and, locate the hands in the img
        nHands = detector.findHands(img, drawHandConnections=False)

        for i in range(nHands):
            img = processHand(img, detector, i)
        print()

        # display the results live on realtime
        cv2.imshow("Fingers Counter", img)
        # Time in ms to wait before frames update (0 => do not update)
        cv2.waitKey(1)
        if cv2.getWindowProperty("Fingers Counter", 4) < 1:
            cap.release()


def processHand(img, detector, handID):
    lm = detector.getLandmarks(handID)  # get the coordinates of the landmarks
    anchor = lm[0]                      # set the wrist landmark as anchor point
    # get the border box of the hand (x, y, width, hight)
    bbox = detector.getBorderBox(handID)
    # get the polar coordinates of the landmarks (angle, distance) [wrist point is (0,0)]
    polarFingers = detector.getPolarFingersLandmarks(handID)
    # count the number of fingers in the image
    number = countFingers(polarFingers)
    # set a scalling factor
    sizeFactor = 5 * (bbox[2] * bbox[3]) / (img.shape[0] * img.shape[1])
    # put the number of fingers on the image above the hand using the scalling factor (sizeFactor) to change the size of the text
    cv2.putText(
        img,
        str(number),
        (int(anchor[0] - (sizeFactor * 60)), bbox[1] - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        sizeFactor * 3 + 1.3,
        (0, 0, 255),
        2,
    )
    return img


def countFingers(fingers):
    """count the number of stritch fingers

    Args:
        fingers (dict): dictionary of fingers and their polar coordinates

    Returns:
        int: the number of stritch fingers
    """
    total = 0
    for finger in fingers.values():
        total += fingerIsStretched(finger)
    return total


def fingerIsStretched(finger):
    """check if a certain finger is stretch or not

    Args:
        finger (list): list of polar coordinates

    Returns:
        bool
    """
    thrushold = (
        finger[-1][1] * 0.60
    )  # 60% of the distance between the wrist point and the beginning of the finger
    return (
        finger[0][1] - finger[-1][1] > thrushold
        and abs(
            finger[0][0] - finger[-1][0],
        )
        < 0.22
    )


if __name__ == "__main__":
    main()
