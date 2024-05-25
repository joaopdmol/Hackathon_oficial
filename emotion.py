import cv2
from deepface import DeepFace
import time
import csv
import pandas as pd
import matplotlib

# Load face cascade classifier
cascade_path = 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(cascade_path)

if face_cascade.empty():
    raise IOError(f"Error loading cascade classifier from {cascade_path}")

# Start capturing video
cap = cv2.VideoCapture(0)

# Dictionary to store the time spent on each emotion
emotion_times = {}
# Dictionary to store the time spent with each count of people
people_times = {}
# Variable to store the start time of the current emotion
start_time = None
# Variable to store the start time of the current people count
people_start_time = time.time()
# Variable to store the current emotion
current_emotion = None
# Variables to track the number of people analyzed
people_count = 0
max_people = 0

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Convert frame to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Convert grayscale frame to RGB format
    rgb_frame = cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2RGB)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Update the number of people being analyzed
    people_count = len(faces)
    if people_count > max_people:
        max_people = people_count

    for (x, y, w, h) in faces:
        # Extract the face ROI (Region of Interest)
        face_roi = rgb_frame[y:y + h, x:x + w]

        # Perform emotion analysis on the face ROI
        result = DeepFace.analyze(face_roi, actions=['emotion'], enforce_detection=False)

        # Determine the dominant emotion
        emotion = result[0]['dominant_emotion']

        # Update the emotion time tracking
        if current_emotion is None:
            current_emotion = emotion
            start_time = time.time()
        elif current_emotion != emotion:
            end_time = time.time()
            elapsed_time = end_time - start_time
            if current_emotion in emotion_times:
                emotion_times[current_emotion] += elapsed_time
            else:
                emotion_times[current_emotion] = elapsed_time
            current_emotion = emotion
            start_time = time.time()
        else:
            # Update the end time for the current emotion
            end_time = time.time()

        # Draw rectangle around face and label with predicted emotion
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.putText(frame, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    # Update the people time tracking
    current_time = time.time()
    if people_count in people_times:
        people_times[people_count] += current_time - people_start_time
    else:
        people_times[people_count] = current_time - people_start_time
    people_start_time = current_time

    # Display the resulting frame with people count
    cv2.putText(frame, f'People: {people_count}', (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('Real-time Emotion Detection', frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        # Update the last emotion time before exiting
        if current_emotion is not None:
            elapsed_time = time.time() - start_time
            if current_emotion in emotion_times:
                emotion_times[current_emotion] += elapsed_time
            else:
                emotion_times[current_emotion] = elapsed_time
        break

# Release the capture and close all windows
cap.release()
cv2.destroyAllWindows()

# Save the total time spent on each emotion and max people count to a CSV file
with open('emotion_times.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Emotion', 'Time (seconds)'])
    for emotion, total_time in emotion_times.items():
        writer.writerow([emotion, total_time])
        print(f"{emotion}: {total_time:.2f} seconds")
    writer.writerow([])
    print(f"Maior numero de pessoas que estavam prestando atencao na aula ao mesmo tempo: {max_people}")

# Save the total time spent with each count of people to a CSV file
with open('people_times.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['People Count', 'Time (seconds)'])
    for count, total_time in people_times.items():
        writer.writerow([count, total_time])
        print(f" {count} Alunos Prestando Atenção: {total_time:.2f} seconds")
