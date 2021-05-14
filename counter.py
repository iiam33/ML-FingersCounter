import HandTrackingModule as htm
import cv2


def main():
    cap = cv2.VideoCapture(0)
    detector = htm.HandDetector()

    while cap.isOpened():

        success, img = cap.read()
        img = cv2.flip(img, 1)
        nHands = detector.findHands(img, draw=False)

        for i in range(nHands):

            lm = detector.getLandmarks(i)
            anchor = lm[0]
            bbox = detector.getBorderBox(i)
            polarFingers = detector.getPolerFingersLandmarks()
            number = countFingers(polarFingers)
            sizeFactor = 5 * (bbox[2] * bbox[3]) / (img.shape[0] * img.shape[1])
            cv2.putText(
                img,
                str(number),
                (int(anchor[0] - (sizeFactor * 60)), bbox[1] - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                sizeFactor * 3 + 1.3,
                (0, 0, 255),
                2,
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
