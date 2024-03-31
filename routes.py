from pathlib import Path
from ultralytics import YOLO

import cv2,os
from ultralytics.utils.plotting import Annotator

model2_path=r"C:\Users\abbes\Downloads\Modelpreparing\model2\crack_spall_segmentation_YOLOv8s_model\segment\train4\weights\best.pt"
model1_path=r"c:\Users\abbes\Downloads\Modelpreparing\model1\crack_segmentation_YOLOv8n_model\train8\weights\best.pt"
model3_path=r"C:\Users\abbes\Downloads\Modelpreparing\model3_YOLOv8S_seg\runs\segment\train3\weights\best.pt"
model4_path=r"C:\Users\abbes\Downloads\model4\runs\segment\train2\weights\best.pt"
image_path=r"C:\Users\abbes\runs\segment\predict8\422473027_24696990526615866_5921879875950736426_n_frames\121.jpg"
images_path=r"C:\Users\abbes\Videos\frames"
video_path=r"C:\Users\abbes\Videos\420712458_25329549856659200_6434361571323618539_n.mp4"
output_path = r"C:\Users\abbes\Videos\frames"
image_croped_path=r"C:\Users\abbes\Videos\frames\crop3"
model=YOLO(model3_path)


noms_classes =  {0: 'crack', 1: 'spall'}
object_data = [] 


def save_object_data(domageID,frame_title,cropped_object_image,frame_number, current_damage_class_id,indice_classe, noms_classe,boxes,confidences):
    # Implement data saving logic with error handling, timestamping, etc.
    # Example using a dictionary:
    object_info = {
           "domageID":domageID,

            "frame_title": frame_title,
            "current_damage_class_id": current_damage_class_id,
            "cropped_object_image":cropped_object_image,
            "frame_number":frame_number,
            "indice_classe": indice_classe,
            "confidences":confidences,
            "noms_classe": noms_classe,
            "boxes": boxes,
        }
   
    object_data.append(object_info)
    # ... (save object_data using your preferred method)




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
nbr_object=0
nb_crack=0
nb_spall=0
num_id=0
# read frames
while ret:
    ret, frame = cap.read()
    
    if ret:
        frame_number+=1
        new_id_assigned = False
        # detect objects
        # track objects
        results = model.track(source=frame, persist=True, conf=0.5, iou=iou, device="cpu",retina_masks=True,max_det=max_det,save_conf=True, imgsz=(height,width))
        #print(results[0])
        
        test_visualis = results[0].plot()


        # visualize
        cv2.imshow('YOLOv8 Tracking', test_visualis)
                # Check if there are any detections
        if results[0].boxes.id is not None:
            # Extract IDs if they exist
            print(results[0].boxes.id)
            
            id_list_detect = results[0].boxes.id.int().tolist()   
            
            if max(id_list_detect) > num_id:# new object(s) detected 

                frame_ = results[0].plot(labels=False)
                max_nbr_id=max(id_list_detect) # peut etre plusieurs object(id) , une sote dans le nbr des id peut se faire expl "1, 2, 3 , 6" , un id deja detecté peut safficher encore dans la list
                print("max id detect",max_nbr_id)

                for indice , id in enumerate(id_list_detect) :
                # Sauvegarder le frame avec un nom de fichier unique



                    if id >num_id :
                        
                        nbr_object+=1
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
                        # annotator.count_labels(counts=nbr_object, count_txt_size=2, color=(255, 255, 255), txt_color=(0, 0, 0))
                        

                        frame_title = f"frame_{frame_number}_object_num_{nbr_object}.jpg"
                        cv2.imwrite(os.path.join(output_path, frame_title), frame_) 



                        objet_decoupe = frame[y_min:y_max, x_min:x_max]
                        cropped_object_image=f"crop_{frame_number}_id{nbr_object}.jpg"
                        cv2.imwrite(os.path.join(image_croped_path, cropped_object_image), objet_decoupe)
                        
                        save_object_data(nbr_object,frame_title,cropped_object_image,frame_number,current_damage_class_id,classe_indice,noms_classe,boxes,confidences)
               

                num_id=max_nbr_id  
              

                
            id_list_detect=None
        
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break       
print("nbr des des id assigné par le track du model:",num_id)
print("nbr reel des objects detectees :",nbr_object)
print("nbr des crack detecté",nb_crack)
print("nbr des spall detecté",nb_spall)

print(object_data)