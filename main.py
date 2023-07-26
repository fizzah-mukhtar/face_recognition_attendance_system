import face_recognition
import cv2
import numpy as np
import csv
from datetime import datetime

video_capture = cv2.VideoCapture(0)

# Load known faces
fizzah_image = face_recognition.load_image_file("faces/fizzah.jpg")
fizzah_encoding = face_recognition.face_encodings(fizzah_image)[0]

fatima_image = face_recognition.load_image_file("faces/fatima.jpg")
fatima_encoding = face_recognition.face_encodings(fatima_image)[0]

known_face_encodings = [fatima_encoding, fizzah_encoding]
known_face_names = ["fatima", "fizzah"]

students = known_face_names.copy()
face_locations = []
face_encodings = []

# get the current date and time
now = datetime.now()
current_date = now.strftime("%Y-%m-%d")

f = open(f"{current_date}.csv", "w+", newline="")
lnwriter = csv.writer(f)

while True:
    _, frame = video_capture.read()
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    # recognize faces
    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        face_distance = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distance)

        if matches[best_match_index]:
            name = known_face_names[best_match_index]
              # add the text if the person is present
        if name in known_face_names:
            font = cv2.FONT_HERSHEY_SIMPLEX
            bottomLeftCornerOfText = (10,100)
            fontscale = 1.5
            fontcolor = (255,0,0)
            thickness = 3
            lineType = 2
            cv2.putText(frame, name + "present", bottomLeftCornerOfText, font, fontscale, fontcolor, thickness, lineType)
            
            if name in students:
                students.remove(name)
                lnwriter.writerow([name, now.strftime("%H:%M:%S")])
            

    cv2.imshow("Attendance", frame)
    if cv2.waitKey(1) & 0xff == ord("q"):
        break

# Don't forget to close the CSV file
f.close()
