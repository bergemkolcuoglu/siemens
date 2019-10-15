import face_recognition
import cv2
from keras.preprocessing import image
import numpy as np
import pyautogui

# Mail göndermek----------------------------------------------------

import time
import smtplib
from datetime import datetime
from email.mime.image import MIMEImage
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

def send_mail(name):

    f_time = datetime.now().strftime('%a %d %b @ %H:%M')
    index = known_face_names.index(name)
    toaddr = known_face_mails[index]
    me = 'ozupythonproject@gmail.com'
    msg = MIMEMultipart()
    msg['Subject'] = f_time
    msg['From'] = me
    msg['To'] = toaddr
    msg.attach(MIMEText("Selam, {0} fotoğraftaki sensin değil mi?".format(name), 'plain'))

    fp = open('C:\\Users\\Pc\\Desktop\\Screenshot.png', 'rb')
    img = MIMEImage(fp.read())
    fp.close()
    msg.attach(img)

    try:
       s = smtplib.SMTP('smtp.gmail.com', 25)
       s.starttls()
       s.login("your_gmail_account@gmail.com","your_password")
       s.send_message(msg)
       s.quit()
       print("Mail successfully sent to "+name)
    except:
       print ("Error: unable to send email")

# Mail gönderme sonu--------------------------------------------------------
       
face_cascade = cv2.CascadeClassifier('D:\\DERSLER\\2_Siemens - STAJ\\haarcascade_frontalface_default.xml')

# Webcam bağlantı
window_name = "Face Recognition App"
video_capture = cv2.VideoCapture(0)


# Duyguları yakalamak için gerekli olanlar
from keras.models import model_from_json
model = model_from_json(open("facial_expression_model_structure.json", "r").read())
model.load_weights('facial_expression_model_weights.h5') #load weights

emotions = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')

# Fotoğrafları tanıtma
bergem_image = face_recognition.load_image_file("D:\\DERSLER\\2_Siemens - STAJ\\foto.png")
bergem_face_encoding = face_recognition.face_encodings(bergem_image)[0]

asim_image = face_recognition.load_image_file("D:\\DERSLER\\2_Siemens - STAJ\\ASM.jpg")
asim_face_encoding = face_recognition.face_encodings(asim_image)[0]

deniz_image = face_recognition.load_image_file("D:\\DERSLER\\2_Siemens - STAJ\\Deniz.jpeg")
deniz_face_encoding = face_recognition.face_encodings(deniz_image)[0]

aysun_image = face_recognition.load_image_file("D:\\DERSLER\\2_Siemens - STAJ\\Aysun.jpg")
aysun_face_encoding = face_recognition.face_encodings(aysun_image)[0]

esra_image = face_recognition.load_image_file("D:\\DERSLER\\2_Siemens - STAJ\\Esra.jpg")
esra_face_encoding = face_recognition.face_encodings(esra_image)[0]

serdal_image = face_recognition.load_image_file("D:\\DERSLER\\2_Siemens - STAJ\\Serdal.jpeg")
serdal_face_encoding = face_recognition.face_encodings(serdal_image)[0]

eray_image = face_recognition.load_image_file("D:\\DERSLER\\2_Siemens - STAJ\\Eray.jpeg")
eray_face_encoding = face_recognition.face_encodings(eray_image)[0]

sedat_image = face_recognition.load_image_file("D:\\DERSLER\\2_Siemens - STAJ\\Sedat.jpeg")
sedat_face_encoding = face_recognition.face_encodings(sedat_image)[0]

fulya_image = face_recognition.load_image_file("D:\\DERSLER\\2_Siemens - STAJ\\Fulya.jpeg")
fulya_face_encoding = face_recognition.face_encodings(fulya_image)[0]

gorkem_image = face_recognition.load_image_file("D:\\DERSLER\\2_Siemens - STAJ\\Gorkem.jpeg")
gorkem_face_encoding = face_recognition.face_encodings(gorkem_image)[0]


# Sıralı encoding bilgileri, sıralı isimler
known_face_encodings = [
    bergem_face_encoding,
    asim_face_encoding,
    deniz_face_encoding,
    aysun_face_encoding,
    esra_face_encoding,
    serdal_face_encoding,
    eray_face_encoding,
    sedat_face_encoding,
    fulya_face_encoding,
    gorkem_face_encoding
]
known_face_names = [
    "Bergem",
    "Asim",
    "Deniz",
    "Aysun",
    "Esra",
    "Serdal",
    "Eray",
    "Sedat",
    "Fulya",
    "Rizeli"
]

known_face_mails = [
        "bergem.kolcuoglux@ozu.edu.tr",
        "asim.doganx@ozu.edu.tr",
        "deniz.ozgunayx@siemens.com",
        "kolcuogluaysunx@gmail.com",
        "esrakolcuoglu68x@gmail.com",
        "serdal.bayramx@siemens.com",
        "eray.yilmazx@siemens.com",
        "sedat.esenx@siemens.com",
        "fulya.durmusx@ozu.edu.tr",
        "tahsin.kaboglux@ozu.edu.tr"
]

# Başlangıç değerlerin girişi
face_locations = []
face_encodings = []
face_names = []
process_this_frame = True

#webcam size büyütmek için
cv2.namedWindow(window_name, cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

while True:
    # Video içinde single frame yakalamak için 
    ret, frame = video_capture.read()
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        #cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
		
        detected_face = frame[int(y):int(y+h), int(x):int(x+w)] 
        detected_face = cv2.cvtColor(detected_face, cv2.COLOR_BGR2GRAY) 
        detected_face = cv2.resize(detected_face, (48, 48)) 
		
        img_pixels = image.img_to_array(detected_face)
        img_pixels = np.expand_dims(img_pixels, axis = 0)		
        img_pixels /= 255 #pixel[0, 1]		
        
        predictions = model.predict(img_pixels) #7 olasılığı ekledik	
        
        # olasılıklardan hangisi fazlaysa onu döndürmek için
        max_index = np.argmax(predictions[0])		
        emotion = emotions[max_index]
		
		# Emotionları yazı olarak ekle
        cv2.putText(frame, emotion, (int(x), int(y)+10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)

    # Daha hızlı yüz tanıma işlemi için videonun karesini 1/4 boyutuna getirme
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

    # Renkleri BGR(openCV)'dan RGB(face_recognition)'ye çevirme.
    rgb_small_frame = small_frame[:, :, ::-1]

    # Her frame'i check etsin
    if process_this_frame:
        
        # Videodaki bütün yüzleri ve encoding hallerini bulmak için
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []
        for face_encoding in face_encodings:
            # Videodaki yüzler bizimkilerle match ediyor mu?
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"
            mail = "Unknown"

            # Videonun içinde know_face_encodings'in içinde match eden yüz varsa döndür
            if True in matches:
                first_match_index = matches.index(True)
                name = known_face_names[first_match_index]
                mail = known_face_mails[first_match_index]

            face_names.append(name)

    process_this_frame = not process_this_frame


    # Sonuçları görüntülemek için
    for (x, y, w, h), name in zip(face_locations, face_names):
        
        # small_frame'de 1/4 tuttuğumuz için normal değerlerine geri döndürüyoruz
        x *= 4
        y *= 4
        w *= 4
        h *= 4

        # Yüzün etrafında kutu çizmek için
        cv2.rectangle(frame, (h-10, x-10), (y+10, w+10), (0, 0, 255), 2)

        # Kutuya label ve yazı eklemek için
        cv2.rectangle(frame, (h - 10, w - 10), (y+10, w+10), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (h , w+5 ), font, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, "{0} tane yuz algilandi!".format(len(face_names)), (10,20), font, 0.5, (0,255,0), 1)


    # Pencereye isim vermek ve görüntülemek
    cv2.imshow(window_name, frame)

    # Pencereyi kapatmak için q'ya basmak gerek
    if cv2.waitKey(1) & 0xFF == ord('q'):
        pyautogui.screenshot('C:\\Users\\Pc\\Desktop\\Screenshot.png')
        break

print("Video içinde son görülen kişiler: ")
for name in face_names:
    print(name)
video_capture.release()
cv2.destroyAllWindows()
for name in face_names:
    send_mail(name)
