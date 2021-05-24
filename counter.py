#!/usr/bin/env python3

import HandTrackingModule as htm
import cv2
import click


@click.command()
@click.option(
    "--max-hands",
    default=2,
    type=int,
    help="Maximum number of hands to be processed at the same time",
)
@click.option("--debug", is_flag=True, help="draw more info for debugging")
def main(max_hands, debug):
    cap = cv2.VideoCapture(0)  # Capture webcam video stream
    detector = htm.HandDetector(maxHands=max_hands)  # Initialize HandDetector

    while cap.isOpened():  # Loop while webcam is open

        _, img = cap.read()  # read images from the webcam
        # mirror the image horizontally (removing this line will not break the program)
        img = cv2.flip(img, 1)
        # process the image to find, detect and, locate the hands in the img
        number_of_hands = detector.find_hands(img, drawHandConnections=debug)

        for i in range(number_of_hands):
            img = process_hand(img, detector, i, debug=debug)

        # display the results live on realtime
        cv2.imshow("Fingers Counter", img)
        # Time in ms to wait before frames update (0 => do not update)
        cv2.waitKey(1)
        if cv2.getWindowProperty("Fingers Counter", 4) < 1:
            cap.release()


def process_hand(img, detector, hand_id, debug=False):
    lm = detector.get_landmarks(hand_id)  # get the coordinates of the landmarks
    anchor = lm[0]  # set the wrist landmark as anchor point
    # get the border box of the hand (x, y, width, height)
    bbox = detector.get_bounding_box(hand_id, drawBox=debug)

    fingers = detector.get_fingers_landmarks(hand_id)
    # get the polar coordinates of the landmarks (angle, distance) [wrist point is (0,0)]
    polar_fingers = detector.get_polar_fingers_landmarks(hand_id)

    # count the number of fingers in the image
    number = 0
    for key, finger in polar_fingers.items():
        number += finger_is_stretched(finger)
        if debug and finger_is_stretched(finger):
            cv2.circle(img, fingers[key][-1], 9, (250, 0, 150), 2)

    # set a scaling factor
    size_factor = (5 * polar_fingers["pinky"][-1][1] / img.shape[0]) ** 2
    # put the number of fingers on the image above the hand
    cv2.putText(
        img,
        str(number),
        (int(anchor[0] - (size_factor * 40)), bbox[1] - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        size_factor * 3.5 + 0.75,
        (0, 0, 255),
        2,
    )
    return img


def count_fingers(fingers):
    """count the number of stretch fingers

            Args  :
    fingers (dict): dictionary of fingers and their polar coordinates

    Returns:
    int    : the number of stretch fingers
    """
    return sum(finger_is_stretched(finger) for finger in fingers.values())


def finger_is_stretched(finger):
    """check if a certain finger is stretch or not

           Args  :
    finger (list): list of polar coordinates

    Returns:
        bool
    """

    # 60% of the distance between the wrist point and the beginning of the finger
    threshold = finger[-1][1] * 0.60

    return tip_is_farther(finger, threshold) and joints_are_aligned(finger)


def joints_are_aligned(finger):
    return abs(finger[0][0] - finger[-1][0]) < 0.22


def tip_is_farther(finger, threshold):
    return finger[0][1] - finger[-1][1] > threshold


if __name__ == "__main__":
    main()
