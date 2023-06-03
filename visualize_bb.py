import cv2 as cv

#Oggetto capture con video
video_path = "[LAB-AI] Matteo Cutroni/Video/20220822_093927.mp4"
cap = cv.VideoCapture(video_path)

#Apertura file txt con tracks
file = open("new_tracks.txt")
Lines = file.readlines()

#VideoWriter per salvare video
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
fourcc = cv.VideoWriter_fourcc(*'mp4v')
out = cv.VideoWriter('video_con_bb.mp4', fourcc, 25.0, (frame_width, frame_height))

while cap.isOpened():
    #Capture frame per frame
    ret, frame = cap.read()
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    #indice del frame attuale
    cap_frame_idx = cap.get(cv.CAP_PROP_POS_FRAMES)

    for box_info in Lines:
        frame_idx, id, left, top, w, h, conf, x,y,z = box_info.split(', ')

        frame_idx, id, left, top, w, h = int(float(frame_idx)), int(float(id)), int(float(left)), int(float(top)), int(float(w)), int(float(h))
        # Disegna il rettangolo delimitatore sul frame se frame attuale è uguale a quello della riga
        if (cap_frame_idx == frame_idx):
            
            frame = cv.rectangle(frame, (left, top), (left+w, top+h), (0, 255, 0), 2)
            frame = cv.putText(frame, f"ID: {id}", (left, top+30), cv.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Mostra il frame con le bounding box
    cv.imshow('Video con bounding box', frame)
    
    #scrive frame nel video di output
    out.write(frame)

    #chiudi con q, pausa con p
    key = cv.waitKey(1)
    if key == ord('q'):
        break
    if key == ord('p'):
        print(cap_frame_idx)
        cv.waitKey(-1) 

cap.release()
out.release()
cv.destroyAllWindows()