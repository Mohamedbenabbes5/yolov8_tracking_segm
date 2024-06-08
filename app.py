from flask import Flask, request,jsonify


from pathlib import Path
from ultralytics import YOLO
import numpy as np
import cv2,os
from ultralytics.utils.plotting import Annotator

import uuid
import time

app = Flask(__name__)


shared = r"C:\Users\abbes\Documents\developpement\BackEndProjectPFE\shared"

damages_images_path =os.path.join(shared,'result') 
cropped_damages_images_path=os.path.join(shared,'result','cropped_images') 
inspectionImages_path = os.path.join(shared, 'uploads', 'inspectionImages')
inspectionVideos_path = os.path.join(shared, 'uploads', 'inspectionVideos')

model=YOLO("best.pt",verbose=False)

dammageLists =  {0: 'crack', 1: 'spall'}

class DamageInfo:
    def __init__(self,  resourceId=None , tracking_id_in_video=None, DetectResultImage=None, croppedDamageImage=None,  videoFrameNumber=None, confidence=None, type=None, bboxe=None,mask=None):
        self.resourceId = resourceId  
        self.tracking_id_in_video = tracking_id_in_video
        
   

        self.DetectResultImage = DetectResultImage
        self.croppedDamageImage = croppedDamageImage

        self.videoFrameNumber = videoFrameNumber
        self.confidence = confidence
        self.type = type
        self.bboxes = bboxe
        self.mask = mask
    
    def to_dict(self):
        return self.__dict__

@app.route('/process_media', methods=['POST'])
def process_media(): 
    try:
        global data_results ,nbr_damage
        data_results = {} 
        nbr_damage=0
               # Vérifier si la requête contient des données JSON
        if request.is_json:
            data_request = request.json.get('media_list')
            print(data_request)
            # Vérifier si la requête contient des clés
            if data_request is not None and len(data_request) > 0:
                for resourceId, file_name in data_request.items():  # Boucler à travers chaque élément de la liste
                    file_extension = os.path.splitext(file_name)[1].lower()
                    print(id, file_name)
                    if file_extension in ('.jpg', '.jpeg', '.png', '.gif','.webp'): 
                        processImage(file_name,resourceId)
                    elif file_extension in ('.mp4', '.avi', '.mov', '.mkv'):
                        processVideo(file_name,resourceId)
                    else:
                        return jsonify({'error': 'Type de fichier non pris en charge'}), 404      
                # Vous pouvez retourner une réponse appropriée une fois que tous les médias ont été traités
                # return jsonify({"message": "Le traitement des médias a été effectué avec succès."}), 200
                return jsonify(data_results), 200

            else:
            # Si la requête ne contient pas de données, renvoyer une erreur appropriée
                return jsonify({"error": "La requête ne contient pas de données JSON."})
        else:
            # Si la requête n'est pas en JSON, renvoyer une erreur appropriée
            return jsonify({"error": "La requête doit être au format JSON."}), 400

    except Exception as e:

    
        app.logger.error(f"Error processing : {e}")
        return jsonify({'error': 'An error occurred while processing the media'}), 500   
    

# @app.route('/process_video', methods=['POST'])
def processVideo(video_name,resourceId):
    try:
        # data_results={}
        global data_results
        global nbr_damage
        global model
        # Récupérer le nom de la vidéo envoyée dans la requête POST
        # video_name = request.json.get('video_name')
        print(video_name)

        # Chemin complet de la vidéo
        video_path = os.path.join(inspectionVideos_path,  video_name)
        if not os.path.exists(video_path):
        # Si le fichier n'existe pas, renvoyer une réponse avec un code d'état 404 et un message d'erreur
           return jsonify({'error_message': 'video not found on the server'}), 404
         # Votre code pour traiter la vidéo ici...
        
        # Retourner une réponse réussie

        cap = cv2.VideoCapture(video_path)

        # # Obtenir la largeur et la hauteur de la vidéo
        # width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        # height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        class_ids=None
        ret = True
        videoFrameNumber=0
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
                videoFrameNumber+=1
                new_id_assigned = False
                # detect damages
                # track damages
                results = model.track(source=frame, persist=True, conf=0.5, iou=0.7, device="cpu",retina_masks=True,max_det=max_det,save_conf=True)
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
                                
                                tracking_id_in_video+=1
                                current_damage_class_id=0
                                boxes=results[0].boxes.xyxy[indice].tolist() 
                                confidence =  round(float(results[0].boxes.conf[indice].item()), 2) 
                                mask=results[0].masks.xy[indice].tolist()   
                                print(mask)
                                classe_indice=int(results[0].boxes.cls[indice].item())                        
                                type=dammageLists[classe_indice]

                                if classe_indice==0:
                                    nb_crack+=1   
                                    current_damage_class_id= nb_crack                    
                                    etiquette=f"Crack {int(confidence*100)}%"
                                    print(f"new {type} detected: {type} num {nb_crack}")

                                else:
                                    nb_spall+=1
                                    current_damage_class_id= nb_spall
                                    etiquette=f"Spall {int(confidence*100)}%"
                                    print(f"new {type} detected: {type} num {nb_spall}")

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
                                
                                croppedDamageImage = frame[y_min:y_max, x_min:x_max]                        
                    
                                frame_title = f"{int(time.time())}_{str(uuid.uuid4())[:8]}.jpg"
                            
                                cv2.imwrite(os.path.join(damages_images_path, frame_title), frame_) 
                                
                                crop_title=f"{int(time.time())}_{str(uuid.uuid4())[:8]}.jpg"
                                cv2.imwrite(os.path.join(cropped_damages_images_path, crop_title), croppedDamageImage)
                                new_damage_info = DamageInfo(
                                    resourceId,
                                    tracking_id_in_video,  
                   
                                    frame_title,
                                    crop_title,
                                    
                                    videoFrameNumber,                        
                                
                                    confidence,
                                    type,
                                    boxes,  
                                    mask                   
                                        )
                                data_results[f'damage{nbr_damage+tracking_id_in_video}'] = new_damage_info.to_dict()
                               
                                
                        num_id=max_nbr_id      
                    id_list_detect=None
                
                if cv2.waitKey(25) & 0xFF == ord('q'):        
                    break   
    
        # return jsonify(data_results), 200
        nbr_damage+=tracking_id_in_video
        model=YOLO("best.pt")
        cv2.destroyAllWindows()  
        cap.release()    
        # Libérer la capture vidéo précédente si elle existe
    except Exception as e:
        app.logger.error(f"Error processing video: {e}")
        return jsonify({'error_message': 'An error occurred while processing the video'}), 500
   
def processImage(image_name,resourceId):
    try:    
        global data_results
        global nbr_damage
        # Récupérer le nom de image envoyée dans la requête POST
        print(image_name)
        print("nbr damages init",nbr_damage)
        # Chemin complet de image
        image_path = os.path.join(inspectionImages_path,image_name)
        if not os.path.exists(image_path):
            # Si le fichier n'existe pas, renvoyer une réponse avec un code d'état 404 et un message d'erreur
            return jsonify({'error_message': 'Image not found on the server'}), 404

        results=model(image_path, conf=0.2,iou=0.9, save=True,project=damages_images_path,name=image_name)
        if (len(results[0].boxes.cls)!=0):
            i=0
            for result in results[0]: 
                i+=1
                print('i',i)
                crop_title= f"{int(time.time())}_{str(uuid.uuid4())[:8]}"
                result.save_crop(cropped_damages_images_path,file_name=crop_title)
    
                data=result[0].boxes  

                classe_indice=int(data.cls.item())   
                bboxe=data.xyxy[0].cpu().tolist()       
    
                confidence = round (float(data.conf.item()),2)
                mask=result.masks.xy[0].tolist()   

                new_damage_info = DamageInfo(
                    resourceId=resourceId,
                    DetectResultImage=image_name,
                    croppedDamageImage=crop_title+".jpg",
                    confidence=confidence,
                    type=dammageLists[classe_indice],
                    bboxe=bboxe,
                    mask=mask
                                            )     
                data_results[f'damage{nbr_damage+i}'] = new_damage_info.to_dict()      
        nbr_damage+=i 
    except Exception as e:
        app.logger.error(f"Error processing image: {e}")
        return jsonify({'error_message': 'An error occurred while processing the image'}), 500


if __name__ == '__main__':
    app.run(debug=True,threaded=True)