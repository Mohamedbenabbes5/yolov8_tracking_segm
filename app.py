from flask import Flask, request, jsonify
import requests
from pathlib import Path
from ultralytics import YOLO
import numpy as np
import cv2,os
from ultralytics.utils.plotting import Annotator


app = Flask(__name__)

image_path=r"C:\Users\abbes\Desktop\369ef49e63f6bc41bc87ad55eb8d479b.jpg"
images_path=r"C:\Users\abbes\Videos\frames"
video_path=r"C:\Users\abbes\Videos\420712458_25329549856659200_6434361571323618539_n.mp4"

output_path = r"C:\Users\abbes\Videos\frames"
image_croped_path=r"C:\Users\abbes\Videos\frames\crop"

def processVideo(video_path,model):
    # model.predict(video_path,save=True,show=True)
    # Ouvrez la vidéo pour la lecture
    cap = cv2.VideoCapture(video_path)

    # Obtenir la largeur et la hauteur de la vidéo
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    class_ids=None
    ret = True
    frame_number=0
    max_det=10
    iou=0.5
    nbr_damage=0
    nb_crack=0
    nb_spall=0
    num_id=0
    # read frames
    noms_classes =  {0: 'crack', 1: 'spall'}
    damage_data = []  
    while ret:
        ret, frame = cap.read()
        
        if ret:
            frame_number+=1
            new_id_assigned = False
            # detect damages
            # track damages
            results = model.track(source=frame, persist=True, conf=0.5, iou=iou, device="cpu",retina_masks=True,max_det=max_det,save_conf=True)
            #print(results[0])
            
            test_visualis = results[0].plot()


            # visualize
            cv2.imshow('YOLOv8 Tracking', test_visualis)
                    # Check if there are any detections
            if results[0].boxes.id is not None:
                # Extract IDs if they exist
                print(results[0].boxes.id)
                
                id_list_detect = results[0].boxes.id.int().tolist()   
                
                if max(id_list_detect) > num_id:# new damage(s) detected 

                    frame_ = results[0].plot(labels=False)
                    max_nbr_id=max(id_list_detect) # peut etre plusieurs damage(id) , une sote dans le nbr des id peut se faire expl "1, 2, 3 , 6" , un id deja detecté peut safficher encore dans la list
                    print("max id detect",max_nbr_id)

                    for indice , id in enumerate(id_list_detect) :
                    # Sauvegarder le frame avec un nom de fichier unique



                        if id >num_id :
                            
                            nbr_damage+=1
                            current_damage_class_id=0
                            boxes=results[0].boxes.xyxy[indice]
                            confidences = float(results[0].boxes.conf[indice].item())
                            mask=results[0].masks.xy[indice]                           
                            classe_indice=int(results[0].boxes.cls[indice].item())                        
                            noms_classe=noms_classes[classe_indice]

                            if classe_indice==0:
                                nb_crack+=1   
                                current_damage_class_id= nb_crack                    
                                etiquette=f"Crack num:{nb_crack} {int(confidences*100)}%"
                                print(f"new {noms_classe} detected: {noms_classe} num {nb_crack}")

                            else:
                                nb_spall+=1
                                current_damage_class_id= nb_spall
                                etiquette=f"Spall num:{nb_spall} {int(confidences*100)}%"
                                print(f"new {noms_classe} detected: {noms_classe} num {nb_spall}")

    


                            x1, y1, x2, y2 =  results[0].boxes.xyxy[indice].cpu().int().tolist()
                        # Marge à ajouter autour des boîtes englobantes (en pixels)
                            marge = 30

                            # Augmenter les boîtes englobantes avec la marge
                            x_min =min(x1,x2)- marge
                            y_min = min(y1,y2)-marge
                            x_max =max(x2,x1)+ marge
                            y_max =max(y1,y2)+ marge

                            image_height, image_width = frame.shape[:2]

                            # Limiter les coordonnées aux dimensions de l'image
                            x_min = max(0, x_min)
                            y_min = max(0, y_min)
                            x_max = min(image_width, x_max)
                            y_max = min(image_height, y_max)

                            annotator=Annotator(im=frame_,font='Arial.ttf',line_width=3)               
                            annotator.seg_bbox(mask, mask_color=(255, 30, 255))
                            annotator.box_label(boxes, label=etiquette, color=(0, 0,255), txt_color=(255, 255, 255), rotated=False)
                            # annotator.count_labels(counts=nbr_damage, count_txt_size=2, color=(255, 255, 255), txt_color=(0, 0, 0))
                            
                            cropped_damage_image = frame[y_min:y_max, x_min:x_max]                        
                
                            damage_info = {
                                "domageID": nbr_damage,
                                "current_damage_class_id": current_damage_class_id,
                                "cropped_damage_image": cropped_damage_image,
                                "damage_image":frame_,
                                "frame_number": frame_number,
                                "indice_classe": classe_indice,
                                "confidences": confidences,
                                "noms_classe": noms_classe,
                                "boxes": boxes
                            }
                            damage_data.append(damage_info)
                    num_id=max_nbr_id  
                

                    
                id_list_detect=None
            
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break       
    return(damage_data) 


def processImage(image_path,model):
    result=model(image_path, conf=0.4)
    numpyImage=result[0].plot()# plot a BGR numpy array of predictions
    return(numpyImage)

        

    
#processVideo(video_path,YOLO("best.pt"))
processImage(image_path,YOLO("best.pt"))
