import cv2
import face_recognition

# Load a known face image and encode it
known_image = face_recognition.load_image_file("known_person.jpg")
known_encoding = face_recognition.face_encodings(known_image)[0]

# Load an unknown image
unknown_image = face_recognition.load_image_file("group_photo.jpg")

# Detect faces and get encodings in the unknown image
face_locations = face_recognition.face_locations(unknown_image)
face_encodings = face_recognition.face_encodings(unknown_image, face_locations)

# Compare each face found in the unknown image to the known face encoding
for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
    matches = face_recognition.compare_faces([known_encoding], face_encoding)
    if matches[0]:
        cv2.rectangle(unknown_image, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(unknown_image, "Known Person", (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

# Display the result
cv2.imshow("Face Recognition", cv2.cvtColor(unknown_image, cv2.COLOR_RGB2BGR))
cv2.waitKey(0)
cv2.destroyAllWindows()
