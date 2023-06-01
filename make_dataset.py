import cv2 as cv
import os

#Apertura file txt con tracks
file = open("[LAB-AI] Matteo Cutroni/Tracks/20220822_093927.txt")
Lines = file.readlines()

    
#ciclo su righe del file
for box_info in Lines:
    frame_idx, id, left, top, w, h, conf, x,y,z = box_info.split(', ')
    frame_idx, id, left, top, w, h = int(float(frame_idx)), int(float(id)), int(float(left)), int(float(top)), int(float(w)), int(float(h))
    dir_path = os.path.join('data', f'{id:04d}')
    if not os.path.isdir(dir_path):
        os.mkdir(dir_path)
    path = os.path.join(dir_path, f'{id:04d}_{frame_idx}')
    crop = cv.imread(f'[LAB-AI] Matteo Cutroni/Video/20220822_093927/{frame_idx:04d}.png')[top:top+h,left:left+w]
    if crop.shape[0] > 10 or crop.shape[1] > 10:
        cv.imwrite(f"{path}.png", crop)
