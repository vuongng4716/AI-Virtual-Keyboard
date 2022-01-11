import cv2
import HandTracking as htm
import numpy as np
import time
import autopy

wcam, hcam = 640, 480
frameR = 100
smoothening = 20
cap = cv2.VideoCapture(0)
cap.set(3, wcam)
cap.set(4, hcam)
pTime = 0
plocX, plocY = 0, 0
clocX, clocY = 0, 0
detector = htm.handDetector(maxHands=1)
wSrc, hSrc = autopy.screen.size()

while True:
    # 1. find hand landmarks
    success, img = cap.read()
    img = detector.findHands(img)
    lmList, bbox = detector.findPosition(img)

    # 2. get the tip of the index and middle fingers
    if len(lmList) != 0:
        x1, y1 = lmList[8][1:]
        x2, y2 = lmList[12][1:]
        #print(x1, y1)
        # 3. check which fingers are up
        fingers = detector.fingersUp()
        #print(fingers)
        # 4. only index finger : Moving Mode
        if fingers[1] == 1 and fingers[2] == 0:
            # 5. Convert coordinates
            cv2.rectangle(img, (frameR, frameR), (wcam-frameR, hcam-frameR),
                          (255, 0, 255), 2)
            x3 = np.interp(x1, (frameR, wcam-frameR), (0, wSrc))
            y3 = np.interp(y1, (frameR, hcam-frameR), (0, hSrc))
            # 6. Smoothen values
            clocX = plocX + (x3 - plocX) / smoothening
            clocY = plocY + (x3 - plocY) / smoothening
            # 7. move Mouse
            autopy.mouse.move(wSrc-clocX, clocY)
            cv2.circle(img, (x1, y1), 15, (255, 0, 255), cv2.FILLED)
            plocX, plocY = clocX, clocY


        # 8. both Index and middle fingers are up : clicking Mode
        if fingers[1] == 1 and fingers[2] == 1:
            # 9. find distance between fingers
            length, img, lineInfo = detector.findDistance(8, 12, img)
            # 10. click mouse if distance short
            if length < 40:
                cv2.circle(img, (lineInfo[4], lineInfo[5]), 15, (0, 255, 0), cv2.FILLED)

                autopy.mouse.click()

            # 11. Frame rate
    cTime = time.time()
    fps = 1/ (cTime - pTime)
    pTime = cTime
    cv2.putText(img, str(int(fps)), (20, 50), cv2.FONT_HERSHEY_PLAIN, 3,
                (255, 0, 0), 3)
    # 12. display
    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break