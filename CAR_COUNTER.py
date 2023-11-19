from ultralytics import  YOLO
import cv2
import cvzone
import math
from sort import *




cap =cv2.VideoCapture("../Videos/Motorway Traffic.mp4")


model = YOLO("../Yolo-Weights/yolov8n.pt")

className = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
           'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
           'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella',
           'handbag', 'tie', 'suitcase', 'frisbee', 'skis','snowboard', 'sports ball', 'kite', 'baseball bat',
           'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
           'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli',
           'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
           'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
           'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
           'teddy bear', 'hair drier', 'toothbrush']

mask = cv2.imread("../IMAGES/mask.png")

#Tracking
tracker = Sort(max_age=10,min_hits=2,iou_threshold=0.3)

limits = [650,400,1050,400]

totalCounts=[]

while True:
    success, img = cap.read()
    print("SUCCESS",success,"IMAGE",img)
    imgRegion = cv2.bitwise_and(img,mask)
    results = model(imgRegion,stream=True)
    detections = np.empty((0, 5))
    for r in results:
        boxes=r.boxes

        for box in boxes:
        #Use anyone of them bounding box
            x1,y1,x2,y2=box.xyxy[0]
            x1, y1, x2, y2=int(x1),int(y1),int(x2),int(y2)
            # print(x1,y1,x2,y2)
            # cv2.rectangle(img,(x1,y1),(x2,y2),(255,0,255),2)

            w, h = x2-x1,y2-y1

            #Confidence
            conf=math.ceil((box.conf[0]*100))/100
            cls = int(box.cls[0])
            print(type(box))
            currentClass = className[cls]
            if (currentClass=="car" or currentClass=="bus" or currentClass=="truck" or
                    currentClass=="motorcycle" and conf>=0.3):
                # cvzone.cornerRect(img, (x1, y1, w, h), l=9,rt=5)
                # cvzone.putTextRect(img, f'{currentClass} {conf}', (max(0, x1), max(35, y1)),
                #                    scale=0.6, thickness=1, offset=3)
                currentArray = np.array([x1,y1,x2,y2,conf])
                detections=np.vstack((detections,currentArray))




    resultsTracker = tracker.update(detections)
    cv2.line(img, (limits[0], limits[1]), (limits[2], limits[3]), (0, 0, 255))

    for result in resultsTracker:
        x1, y1, x2, y2, id = result
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        print(result)
        w, h = x2 - x1, y2 - y1
        cvzone.cornerRect(img, (x1, y1, w, h), l=9,rt=2,colorR=(255,0,0))
        cvzone.putTextRect(img, f' {int(id)}', (max(0, x1), max(35, y1)),
                           scale=2, thickness=2, offset=3)
        cx, cy = x1 + w // 2, y1 + h // 2
        cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)
        if limits[0]<cx< limits[2] and limits[1]<cy<limits[1]+250:
            if totalCounts.count(id)==0:
                totalCounts.append(id)
                cv2.line(img, (limits[0], limits[1]), (limits[2], limits[3]), (0, 255, 0))

    cvzone.putTextRect(img, f' Count :: {len(totalCounts)}', (50, 40),scale=2, thickness=2, offset=3)

    cv2.imshow("Image",img)
    cv2.imshow("ImageRegion", imgRegion)
    cv2.waitKey(2)


