

from pathlib import Path
from ultralytics import YOLO
import numpy as np
import cv2,os
from ultralytics.utils.plotting import Annotator
import json
import gc
shared = r"C:\Users\abbes\Documents\developpement\BackEndProjectPFE\shared"

damages_images_path =os.path.join(shared,'result') 

cropped_damages_images_path=os.path.join(shared,'result','cropped_images') 

model=YOLO("best.pt")

noms_classes =  {0: 'crack', 1: 'spall'}

data_results={}
nbr_damage=0

shared = r"C:\Users\abbes\Documents\developpement\BackEndProjectPFE\shared"

class DamageInfo:
    def __init__(self, tracking_id_in_video=None,sources=None ,current_damage_class_id=None, damage_image=None, cropped_damage_image=None,  frame_number=None, confidences=None, noms_classe=None, bboxe=None,mask=None):
        self.tracking_id_in_video = tracking_id_in_video
        self.sources = sources
        self.current_damage_class_id = current_damage_class_id

        self.damage_image = damage_image
        self.cropped_damage_image = cropped_damage_image

        self.frame_number = frame_number
        self.confidences = confidences
        self.noms_classe = noms_classe
        self.bboxes = bboxe
        self.mask = None
    
    def to_dict(self):
        return self.__dict__


def processImage(image_name):
    try:    
        global data_results

        global nbr_damage
        # Récupérer le nom de image envoyée dans la requête POST
       # image_name = request.json.get('image_name')
        print(image_name)
        print("nbr damages init",nbr_damage)
        # Chemin complet de image
        image_path = os.path.join(shared, 'public', 'uploads', 'images', image_name)
        if not os.path.exists(image_path):
            return "Could not find image"
            # Si le fichier n'existe pas, renvoyer une réponse avec un code d'état 404 et un message d'erreur
      
        results=model(image_path, conf=0.2,iou=0.9, save=True,project=damages_images_path,name=image_name,imgsz=640)
        if (len(results[0].boxes.cls)!=0):
            i=0
            for result in results[0]: 
                i+=1
                print('i',i)
                crop_title= f"crop_{i}_{image_name}"
                result.save_crop(cropped_damages_images_path,file_name=crop_title)
    
                data=result[0].boxes  

                classe_indice=int(data.cls.item())   
                bboxe=data.xyxy[0].cpu().tolist()       
    
                confidences = float(data.conf.item())
                mask=result.masks.xy[0].tolist()   

                new_damage_info = DamageInfo(
                    sources=image_name,
                    damage_image=image_name,
                    cropped_damage_image=crop_title,
                    confidences=confidences,
                    noms_classe=noms_classes[classe_indice],
                    bboxe=bboxe,
                    mask=mask
                                            )
                            
                 
                data_results[f'damage{nbr_damage+i}'] = new_damage_info.to_dict()   
                                       
               
        nbr_damage+=i 
        # return jsonify(data_results), 200
    except Exception as e:
        return e

def processVideo(video_name):
    try:
        # data_results={}
        global data_results
        global nbr_damage
        global model
        # Récupérer le nom de la vidéo envoyée dans la requête POST
        # video_name = request.json.get('video_name')
        print(video_name)

        # Chemin complet de la vidéo
        video_path = os.path.join(shared, 'public', 'uploads', 'videos', video_name)
        if not os.path.exists(video_path):
        # Si le fichier n'existe pas, renvoyer une réponse avec un code d'état 404 et un message d'erreur
           return "video not found"
         # Votre code pour traiter la vidéo ici...
        
        # Retourner une réponse réussie
        cv2.destroyAllWindows()
        

        cap = cv2.VideoCapture(video_path)

        # # Obtenir la largeur et la hauteur de la vidéo
        # width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        # height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        
        class_ids=None
        ret = True
        frame_number=0
        max_det=10
        iou=0.5
        tracking_id_in_video=0
        nb_crack=0
        nb_spall=0
        num_id=0
        # read frames
        while ret:
            ret, frame = cap.read()
            
            if ret:
                frame_number+=1
                new_id_assigned = False
                # detect damages
                # track damages
                results = model.track(source=frame, persist=True, conf=0.5, iou=iou, device="cpu",retina_masks=True,max_det=max_det,save_conf=True ,imgsz=640)
                #print(results[0])
                
                test_visualis = results[0].plot()


                # visualize
                # cv2.imshow('YOLOv8 Tracking', test_visualis)
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
                                
                                tracking_id_in_video+=1
                                current_damage_class_id=0
                                boxes=results[0].boxes.xyxy[indice].tolist() 
                                confidences = float(results[0].boxes.conf[indice].item())
                                mask=results[0].masks.xy[indice].tolist()   
                                print(mask)
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
                                # annotator.count_labels(counts=tracking_id_in_video, count_txt_size=2, color=(255, 255, 255), txt_color=(0, 0, 0))
                                
                                cropped_damage_image = frame[y_min:y_max, x_min:x_max]                        
                    
                                frame_title = f"frame_{frame_number}_damage_num_{tracking_id_in_video}.jpg"
                            
                                cv2.imwrite(os.path.join(damages_images_path, frame_title), frame_) 
                                
                                crop_title=f"crop_{frame_number}_id{tracking_id_in_video}.jpg"
                                cv2.imwrite(os.path.join(cropped_damages_images_path, crop_title), cropped_damage_image)


                                new_damage_info = DamageInfo(
                                    tracking_id_in_video,
                                    video_name,      
                                    current_damage_class_id,
                                    frame_title,
                                    crop_title,
                                    
                                    frame_number,                        
                                
                                    confidences,
                                    noms_classe,
                                    boxes,  
                                    mask                   
                                        )
                        


                            

                                data_results[f'damage{nbr_damage+tracking_id_in_video}'] = new_damage_info.to_dict()
                               
                                
                        num_id=max_nbr_id  
                    

                        
                    id_list_detect=None
                
                if cv2.waitKey(25) & 0xFF == ord('q'):
                    break   
        model=YOLO("best.pt")

        # return jsonify(data_results), 200
        nbr_damage+=tracking_id_in_video
        cap.release() 
        if 'cap' in locals() and cap.isOpened():
            cap.release()
    except Exception as e:
    
        print(e)

# processVideo(video_path,YOLO("best.pt"))
processVideo("vid1.mp4")

cv2.destroyAllWindows()

processVideo("vid3.mp4")


# processImage( "bbb.jpg")

# processImage( "2.jpg")
# #processVideo("vid2.mp4")
# processImage( "1.jpg")
print("nbr_damage",nbr_damage)
print("res",data_results)
