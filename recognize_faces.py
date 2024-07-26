import cv2
import pickle

# Path to the Haar Cascade file
cascade_path = "data/haarcascade_frontalface_default.xml"
facedetect = cv2.CascadeClassifier(cascade_path)

# Check if the classifier loaded correctly
if facedetect.empty():
    print(f"Error loading cascade classifier from {cascade_path}")
    exit()

# Load the trained face recognizer
face_recognizer = cv2.face.LBPHFaceRecognizer_create()
face_recognizer.read('data/face_recognizer.yml')

# Load the label map
with open('data/label_map.pkl', 'rb') as f:
    label_map = pickle.load(f)

# Reverse the label map
reverse_label_map = {v: k for k, v in label_map.items()}

# Initialize video capture
video = cv2.VideoCapture(0)

while True:
    ret, frame = video.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = facedetect.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        face_img = gray[y:y + h, x:x + w]
        resized_img = cv2.resize(face_img, (50, 50))
        label, confidence = face_recognizer.predict(resized_img)

        name = reverse_label_map[label]

        cv2.putText(frame, name, (x, y - 10), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.imshow("Recognizing Faces", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()
