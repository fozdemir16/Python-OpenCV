import cv2
import numpy as np

video = cv2.VideoCapture("video1.mp4") #Seçtiğimiz video üzerinde çalışma yapıyoruz
insan_bulucu = cv2.CascadeClassifier("haarcascade_fullbody.xml") #İnsan tanıma için gerekli xml dosyası


while True:
    ret,kare = video.read() #ret objesi ve numpy sınıfından bir obje
    griton = cv2.cvtColor(kare,cv2.COLOR_BGR2GRAY)
    beden = insan_bulucu.detectMultiScale(griton,1.1,3) # İnsan bedenlerine ait dikdörtgenler

    for(x,y,w,h) in beden:
        cv2.rectangle(kare,(x,y),(x+w,y+h),(255,0,0),3) # İnsan bedenlerini mavi kare içine alır

    cv2.imshow("video",kare)
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()
