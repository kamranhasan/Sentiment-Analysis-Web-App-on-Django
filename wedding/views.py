from django.conf import settings
from django.shortcuts import render,redirect 
# from guests.save_the_date import SAVE_THE_DATE_CONTEXT_MAP
from .models import Document
from django.core.files.storage import FileSystemStorage
from .forms import DocumentForm

# from facial_emotion_image import emotional
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import imutils
import cv2
import numpy as np
import sys

def home(request):
    documents = Document.objects.all()
    return render(request, 'home.html', { 'documents': documents })

def simple_upload(request):
    if request.method == 'POST' and request.FILES['myfile']:
        # haha=emotional()
        myfile = request.FILES['myfile']
        fs = FileSystemStorage()
        filename = fs.save(myfile.name, myfile)
        uploaded_file_url = fs.url(filename)
        return render(request, 'home.html', {
            'uploaded_file_url': uploaded_file_url
        })
        
    return render(request, 'home.html')


def img_submit(request):
    url = request.POST.dict()
    return HttpResponse(url)



# def emotional(request):
#     # parameters for loading data and images
#     detection_model_path = 'haarcascade_files/haarcascade_frontalface_default.xml'
#     emotion_model_path = 'models/_mini_XCEPTION.106-0.65.hdf5'

#     vidcap = cv2.VideoCapture('download.mp4')
#     success,image = vidcap.read()
#     count = 0
#     success = True
#     while success:
#         cv2.imwrite("frame%d.jpg" % count, image)     # save frame as JPEG file
#         success,image = vidcap.read()
#         count += 1
#     emotions=[]
#     for i in range(0,150,10):
#         a="frame"+str(i)
#         a=a+".jpg"
#         img_path = a

#         # hyper-parameters for bounding boxes shape
#         # loading models
#         face_detection = cv2.CascadeClassifier(detection_model_path)
#         emotion_classifier = load_model(emotion_model_path, compile=False)
#         EMOTIONS = ["angry","disgust","scared", "happy", "sad", "surprised","neutral"]


#         #reading the frame
#         orig_frame = cv2.imread(img_path) 
#         frame = cv2.imread(img_path,0)
#         faces = face_detection.detectMultiScale(frame,scaleFactor=1.1,minNeighbors=5,minSize=(30,30),flags=cv2.CASCADE_SCALE_IMAGE)

#         if len(faces) > 0:
#             faces = sorted(faces, reverse=True,key=lambda x: (x[2] - x[0]) * (x[3] - x[1]))[0]
#             (fX, fY, fW, fH) = faces
#             roi = frame[fY:fY + fH, fX:fX + fW]
#             roi = cv2.resize(roi, (48, 48))
#             roi = roi.astype("float") / 255.0
#             roi = img_to_array(roi)
#             roi = np.expand_dims(roi, axis=0)
#             preds = emotion_classifier.predict(roi)[0]
#             emotion_probability = np.max(preds)
#             label = EMOTIONS[preds.argmax()]
#             emotions.append(label)
#             cv2.putText(orig_frame, label, (fX, fY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
#             cv2.rectangle(orig_frame, (fX, fY), (fX + fW, fY + fH),(0, 0, 255), 2)
            
#         # cv2.imshow('test_face', orig_frame)
#         cv2.imwrite('test_output/'+img_path.split('/')[-1],orig_frame)
#         if (cv2.waitKey(2000) & 0xFF == ord('q')):
#             sys.exit("Thanks")
#         cv2.destroyAllWindows()
#     angry=disgust=scared=happy=surprised=sad=neutral=0
#     haha={'angry':0,'disgust':0,'scared':0,'happy':0,'surprised':0,'sad':0,'neutral':0}
#     for i in range(len(emotions)):
#         if emotions[i]=="angry":
#             angry+=1
#         elif emotions[i]=="disgust":
#             disgust+=1
#         elif emotions[i]=="scared":
#             scared+=1
#         elif emotions[i]=="happy":
#             happy+=1
#         elif emotions[i]=="surprised":
#             surprised+=1
#         elif emotions[i]=="sad":
#             sad+=1
#         elif emotions[i]=="neutral":
#             neutral+=1
#         haha[angry]=angry
#         haha[disgust]=disgust
#         haha[scared]=scared
#         haha[happy]=happy
#         haha[surprised]=surprised
#         haha[sad]=sad
#         haha[neutral]=neutral
#         return render(request, 'upload.html',context=[haha])









# default_storage.exists("/bigday/media/video4.mp4")
# def normalupload(request):
#     if request.method == 'POST' and request.FILES['myfile']:
#         myfile = request.FILES['file']
#         fs =FileSystemStorage()
#         filename = fs.save(file.name,file)
#         url =fs.url(filename)
#         new_video =Video(
#             videoupload = url
#         )
#         new_video.save()
#         return redirect('/home/')
#     else:
#         pass