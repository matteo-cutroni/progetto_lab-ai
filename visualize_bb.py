import cv2 as cv

#Oggetto capture con nostro video
video_path = "[LAB-AI] Matteo Cutroni/Video/20220822_093927.mp4"
cap = cv.VideoCapture(video_path)

#Apertura file txt con tracks
file = open("[LAB-AI] Matteo Cutroni/Tracks/20220822_093927.txt")
Lines = file.readlines()

# frame_width = int(cap.get(3))
# frame_height = int(cap.get(4))
# fourcc = cv.VideoWriter_fourcc(*'MJPG')
# out = cv.VideoWriter('output.mp4', fourcc, 25.0, (frame_width, frame_height))

while cap.isOpened():
    #Capture frame per frame
    ret, frame = cap.read()
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    cap_frame_idx = cap.get(cv.CAP_PROP_POS_FRAMES)
    # Leggi le informazioni sulla bounding box dal file MOT per il frame corrente
    # e disegna la bounding box sul frame utilizzando OpenCV
    # Assumendo che le informazioni sulle bounding box siano presenti nella variabile 'box_info'
    for box_info in Lines:
        frame_idx, id, left, top, w, h, conf, x,y,z = box_info.split(', ')

        frame_idx, id, left, top, w, h = int(float(frame_idx)), int(float(id)), int(float(left)), int(float(top)), int(float(w)), int(float(h))
        # Disegna il rettangolo delimitatore sul frame
        if (cap_frame_idx == frame_idx):
            
            frame = cv.rectangle(frame, (left, top), (left+w, top+h), (0, 255, 0), 2)
            frame = cv.putText(frame, f"ID: {id}", (left, top-10), cv.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        #out.write(frame)
    # Mostra il frame con le bounding box
    cv.imshow('Video con bounding box', frame)
    if cv.waitKey(1) == ord('q'):
        break

cap.release()
cv.destroyAllWindows()