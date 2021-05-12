import HandTrackingModule as htm
import cv2


def main():
    cap = cv2.VideoCapture(0)
    detector = htm.HandDetector()

    while cap.isOpened():

        success, img = cap.read()
        img = cv2.flip(img, 1)
        nHands = detector.findHands(img)

        for i in range(nHands):

            lm = detector.getLandmarks(i)
            anchor = lm[0]
            bbox = detector.getBoarderBox(i)
            polarFingers = detector.getPolerFingersLandmarks()

            number = countFingers(polarFingers)

            cv2.putText(
                img,
                str(number),
                (anchor[0] - 28, bbox[1] - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                3,
                (255, 0, 255),
                3,
            )

        cv2.imshow("Fingers Counter", img)
        cv2.waitKey(1)
        if cv2.getWindowProperty("Fingers Counter", 4) < 1:
            cap.release()


def fingerIsStretched(finger):
    thrushold = finger[-1][1] * 0.60
    return (
        finger[0][1] - finger[-1][1] > thrushold
        and abs(
            finger[0][0] - finger[-1][0],
        )
        < 0.22
    )


def countFingers(fingers):
    total = 0
    for finger in fingers.values():
        total += fingerIsStretched(finger)
    return total


if __name__ == "__main__":
    main()
