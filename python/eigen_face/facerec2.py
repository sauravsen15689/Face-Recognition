
import cv2, sys,  os
import numpy as np

size = 4
fn_haar = 'haarcascade_frontalface_default.xml'
fn_dir = 'att_faces'
# Part 2: Use fisherRecognizer on camera stream
model = cv2.face.createEigenFaceRecognizer()
model.load('trainer.yml')

(im_width, im_height) = (112, 92)
(images, lables, names, id) = ([], [], {}, 0)
for (subdirs, dirs, files) in os.walk(fn_dir):
    for subdir in dirs:
        names[id] = subdir
        subjectpath = os.path.join(fn_dir, subdir)
        for filename in os.listdir(subjectpath):
            path = subjectpath + '/' + filename
            lable = id
            images.append(cv2.imread(path, 0))
            lables.append(int(lable))
        id += 1


# Create a Numpy array from the two lists above
(images, lables) = [np.array(lis) for lis in [images, lables]]

haar_cascade = cv2.CascadeClassifier(fn_haar)
webcam = cv2.VideoCapture(0)
while True:
    (rval, frame) = webcam.read()
    #frame = cv2.flip(frame, 1, 0)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    mini = cv2.resize(gray, (gray.shape[1] // size, gray.shape[0] // size))
    faces = haar_cascade.detectMultiScale(mini)
    for i in range(len(faces)):
        face_i = faces[i]
        (x, y, w, h) = [v * size for v in face_i]
        face = gray[y:y + h, x:x + w]
        face_resize = cv2.resize(face, (im_width, im_height))
        prediction = model.predict(np.asarray(face_resize))
        print(prediction)
        # Try to recognize the face
        #print(type(names))
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Write the name of recognized face
        # [1]
        #if int(prediction[1]) < 10:
        if prediction !=-1:
            cv2.putText(frame,'%s' % (names[prediction]),
            (x - 10, y - 10), cv2.FONT_HERSHEY_PLAIN,1,(0, 0, 255), 2)

            #cv2.putText(frame, '%s' % (names[prediction]), (x - 10, y - 10), cv2.FONT_HERSHEY_PLAIN, 2,(0, 0, 255), 2)
        else:
            cv2.putText(frame, 'Unknown', (x - 10, y - 10), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)

    cv2.imshow('OpenCV', frame)
    key = cv2.waitKey(1)
    if key == 27:
        break
