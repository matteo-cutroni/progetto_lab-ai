import cv2 as cv

frame_path = "Test_set/Grapes_001/0300.png"
frame = cv.imread(frame_path)

#Apertura file txt con tracks
file = open("Test_set/gt.txt")
Lines = file.readlines()

for box_info in Lines:
    frame_idx, id, left, top, w, h, conf, x,y = box_info.split(',')

    frame_idx, id, left, top, w, h = int(float(frame_idx)), int(float(id)), int(float(left)), int(float(top)), int(float(w)), int(float(h))
    # Disegna il rettangolo delimitatore sul frame se frame attuale è uguale a quello della riga
    if (300 == frame_idx):
        
        frame = cv.rectangle(frame, (left, top), (left+w, top+h), (0, 255, 0), 3)
        if (id == 8):
            frame = cv.putText(frame, f"ID: {id}", (left+5, top+38), cv.FONT_HERSHEY_DUPLEX, 1.5, (0, 255, 0), 2)
        else:
            frame = cv.putText(frame, f"ID: {id}", (left, top-10), cv.FONT_HERSHEY_DUPLEX, 1.5, (0, 255, 0), 2)
#salva immagine
cv.imwrite('0300_con_bb.png', frame)

cv.destroyAllWindows()