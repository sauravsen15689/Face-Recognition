#!/usr/bin/env python

import random
import socket, select
from time import gmtime, strftime
from random import randint


import cv2
import sys
from time import sleep

cascPath = "lbpcascade_frontalface.xml"
faceCascade = cv2.CascadeClassifier(cascPath)

video_capture = cv2.VideoCapture(0)

if not video_capture.isOpened():
 print('Unable to load camera.')
 sleep(5)
 pass

    # Capture frame-by-frame
ret, frame = video_capture.read()

gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )

    # Draw a rectangle around the faces
for (x, y, w, h) in faces:
 cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # Display the resulting frame
cv2.imshow('Video', frame)


    # Display the resulting frame
cv2.imshow('Video', frame)
cv2.imwrite('img.png', frame)

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()
sleep(10)

image = "img.png"

HOST = '127.0.0.1'
PORT = 6666

sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_address = (HOST, PORT)
sock.connect(server_address)

try:

    # open image
    myfile = open(image, 'rb')
    bytes = myfile.read()
    size = len(bytes)

    # send image size to server
    sock.sendall("SIZE %s" % size)
    answer = sock.recv(4096)

    print 'answer = %s' % answer

    # send image to server
    if answer == 'GOT SIZE':
        sock.sendall(bytes)

        # check what server send
        answer = sock.recv(4096)
        print 'answer = %s' % answer

        if answer == 'GOT IMAGE' :
            sock.sendall("BYE BYE ")
            print 'Image successfully send to server'

    myfile.close()

finally:
    sock.close()
