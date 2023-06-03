import cv2 as cv
import os
from tqdm import tqdm


def skim_tracks(old_tracks, new_tracks, min_tracks):
    
    #apre tracks originali
    file = open(old_tracks)
    lines = file.readlines()

    count = {}
    new_lines = []

    #dizionario per contare istanze di ogni id
    for line in lines:
        id = line.split(', ')[1]
        if id in count:
            count[id] += 1
        else:
            count[id] = 1

    #se ci sono più di min_tracks istanze aggiunge la riga alle nuove righe
    for line in lines:
        id = line.split(', ')[1]
        if count[id] > min_tracks:
            new_lines.append(line)

    #crea nuovo file e scrive le nuove righe
    file = open(new_tracks, 'w')
    file.writelines(new_lines)


def create_data(tracks, frames_folder):

    #apertura file txt con tracks
    file = open(tracks)
    lines = file.readlines()

    with tqdm(total=len(lines), desc='Progresso', unit='righe') as progress_bar:
        #ciclo su righe del file
        for box_info in lines:
            frame_idx, id, left, top, w, h, conf, x,y,z = box_info.split(', ')
            frame_idx, id, left, top, w, h = int(float(frame_idx)), int(float(id)), int(float(left)), int(float(top)), int(float(w)), int(float(h))

            #crea directory se non esiste già
            dir_path = os.path.join('data', f'{id:04d}')
            if not os.path.isdir(dir_path):
                os.mkdir(dir_path)

            #path del crop
            path = os.path.join(dir_path, f'{id:04d}_{frame_idx}.png')

            #legge frame e fa il crop
            crop = cv.imread(f'{frames_folder}/{frame_idx-1:04d}.png')[top:top+h,left:left+w]

            #non considera immagini troppo piccole
            if crop.shape[0] > 20 and crop.shape[1] > 20:
                cv.imwrite(f"{path}", crop)

            progress_bar.update(1)


old_tracks = "[LAB-AI] Matteo Cutroni/Tracks/20220822_093927.txt"
new_tracks = "new_tracks.txt"
frames_folder = "[LAB-AI] Matteo Cutroni/Video/20220822_093927"
min_tracks = 50


skim_tracks(old_tracks, new_tracks, min_tracks)
create_data(new_tracks, frames_folder)