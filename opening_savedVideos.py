import cv2

cap = cv2.VideoCapture("event_clips/clip_20260619_203116.avi")

if not cap.isOpened():
    print("❌ Could not open video file")
    exit()

while True:
    ret, frame = cap.read()

    if not ret:
        print("End of video")
        break

    cv2.imshow("Video playback", frame)

    if cv2.waitKey(30) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()