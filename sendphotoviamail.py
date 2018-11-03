# from client import jasperpath
# from os.path import expanduse
import smtplib
from datetime import datetime

import cv2

WORDS = ["SELFIE", "CAMERA", "CAPTURE", "PHOTO"]

gmail_user = ""
gmail_password = ""


def handle(text, mic, profile):
    home = expanduser("~")
    WINDOW_NAME = "Face Detective"

    cascPath = jasperpath.data('cascade', 'haarcascade_frontalface_default.xml')
    faceCascade = cv2.CascadeClassifier(cascPath)

    video_capture = cv2.VideoCapture(0)
    cv2.namedWindow(WINDOW_NAME, cv2.CV_WINDOW_AUTOSIZE)

    while True:

        ret, frame = video_capture.read()

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.3,
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv2.cv.CV_HAAR_SCALE_IMAGE
        )

        mic.say("Number Of Faces Detected {0}!".format(len(faces)))

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        cv2.startWindowThread()
        cv2.imshow(WINDOW_NAME, frame)

        k = cv2.waitKey(1)
        if k == 27:
            break
        elif len(faces) > 0:
            photoname = home + "/out/%s.png" % datetime.now().strftime("%Y%m%d-%H%M%S")
            mic.say(photoname)
            cv2.imwrite(photoname, frame)
            server = smtplib.SMTP_SSL('smtp.gmail.com', 465)
            server.ehlo()
            server.login(gmail_user, gmail_password)
            # server.sendmail(sent_from, to, email_text)
            server.close()

            break

    video_capture.release()
    cv2.waitKey(1)
    cv2.destroyAllWindows()
    cv2.waitKey(1)
    mic.say("Face detected")


def isValid(text):
    return any(word in text.upper() for word in WORDS)
