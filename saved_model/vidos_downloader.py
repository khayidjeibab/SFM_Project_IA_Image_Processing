import cv2

# Liste des objets que vous voulez détecter
objects_of_interest = ["climatiseure", "Laptop", "Microwave", "Refrigeratueure"]

####### From Image #######
def ImgFile():
   img = cv2.imread('person.png')

   classNames = []
   classFile = 'coco.names'

   with open(classFile, 'rt') as f:
      classNames = f.read().rstrip('\n').split('\n')

   configPath = 'labelmap.pbtxt'
   weightpath = 'saved_model.pb'

   net = cv2.dnn_DetectionModel(weightpath, configPath)
   net.setInputSize(320, 230)
   net.setInputScale(1.0 / 127.5)
   net.setInputMean((127.5, 127.5, 127.5))
   net.setInputSwapRB(True)

   classIds, confs, bbox = net.detect(img, confThreshold=0.5)
   print(classIds, bbox)

   for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):
      label = classNames[classId - 1]
      if label in objects_of_interest:
         cv2.rectangle(img, box, color=(0, 255, 0), thickness=2)
         cv2.putText(img, label, (box[0] + 10, box[1] + 20), 
                     cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), thickness=2)

   cv2.imshow('Output', img)
   cv2.waitKey(0)
######################################

####### From Video File #######
import cv2

def VideoFile(video_path):
    cam = cv2.VideoCapture(video_path)

    classNames = []
    classFile = 'coco.names'

    with open(classFile, 'rt') as f:
        classNames = f.read().rstrip('\n').split('\n')

   configPath = 'Config.pbtxt'
   weightpath = 'saved_model.pb'

    net = cv2.dnn.DetectionModel(weightpath, configPath)
    net.setInputSize(320, 230)
    net.setInputScale(1.0 / 127.5)
    net.setInputMean((127.5, 127.5, 127.5))
    net.setInputSwapRB(True)

    # Create a named window and resize it
    cv2.namedWindow('Output', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Output', 430, 430)

    while True:
        success, img = cam.read()
        if not success:
            break
        classIds, confs, bbox = net.detect(img, confThreshold=0.5)
        print(classIds, bbox)

        if len(classIds) != 0:
            for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):
                label = classNames[classId - 1]
                if label in objects_of_interest:
                    cv2.rectangle(img, box, color=(0, 255, 0), thickness=2)
                    cv2.putText(img, label, (box[0] + 10, box[1] + 20), 
                                cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), thickness=2)

        cv2.imshow('Output', img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cam.release()
    cv2.destroyAllWindows()

# Set the video path and objects of interest
video_path = 'path_to_your_video.mp4'
objects_of_interest = ["climatiseure", "Laptop", "Microwave", "Refrigeratueure"] # Example objects of interest

# Run the function
VideoFile(video_path)


######################################

## Call ImgFile() Function for Image, Camera() Function for Webcam, or VideoFile() Function for Video
# ImgFile()
# Camera()
VideoFile('Video.mp4')
