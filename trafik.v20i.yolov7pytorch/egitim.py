# Model Eğitimi
"""from ultralytics import YOLO

# Modeli yükleyin (önceden eğitilmiş bir model veya sıfırdan eğitim)
model = YOLO("yolov8n.pt")

# Eğitim verisini belirtin
model.train(data="data.yaml", epochs=50)
"""

# Gerekli kütüphaneler import ediliyor
from flask import Flask, Response, render_template, request, redirect, url_for, flash
from ultralytics import YOLO
import cv2
from gtts import gTTS
import pygame
import time
import threading
from io import BytesIO
import os
from werkzeug.utils import secure_filename

# Flask uygulaması ve konfigürasyon ayarları
app = Flask(__name__)  # Flask uygulamasını başlatıyoruz
app.secret_key = 'güvenli_bir_anahtar_buraya'  # Uygulamanın güvenliği için bir secret key
app.config['UPLOAD_FOLDER'] = 'static/uploads'  # Yüklenen dosyaların kaydedileceği klasör
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif'}  # Kabul edilen dosya uzantıları
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # Maksimum dosya boyutu: 16MB

# Model yükleme: YOLO modelini kullanarak trafik işaretlerini tespit etmeye başlıyoruz
model = YOLO("D:/3_sinif_bahar/Uygulama_Tasarimi/trafik.v20i.yolov7pytorch/runs/detect/train/weights/best.pt")

# Web kamerasını başlatıyoruz (0, default kamera olarak kabul edilir)
cap = cv2.VideoCapture(0)

# Ses ayarları: pygame ile ses işlemleri yapılacak
pygame.mixer.init()  # Pygame ses modülünü başlatıyoruz
speech_lock = threading.Lock()  # Sesli yanıtlar için thread kilidi
last_detected = {}  # En son tespit edilen nesneler için kayıt
CONFIDENCE_THRESHOLD = 0.85  # Nesne tespiti için güven eşiği
MIN_SPEECH_INTERVAL = 4.0  # Sesli yanıtlar arasında minimum zaman aralığı (saniye)

# Yükleme klasörü oluşturuluyor (varsa oluşturulmaz)
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Dosya uzantısı kontrol fonksiyonu
def allowed_file(filename):
    """Dosya uzantısı kontrolü"""
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

# Metinleri sesli hale getiren fonksiyon
def speak(text):
    """Metni seslendirme fonksiyonu"""

    # _speak() fonksiyonu bir iş parçacığında çalıştırılacak
    def _speak():
        try:
            tts = gTTS(text=text, lang='tr', slow=False)  # gTTS ile metni sesli hale getiriyoruz
            audio_bytes = BytesIO()  # Ses verisini hafızada saklamak için byte akışı oluşturuyoruz
            tts.write_to_fp(audio_bytes)  # Metni ses dosyasına yazıyoruz
            audio_bytes.seek(0)  # Byte akışını başa alıyoruz

            # Ses çalınırken aynı anda başka bir sesin çalınmaması için kilit kullanıyoruz
            with speech_lock:
                pygame.mixer.music.load(audio_bytes, 'mp3')  # MP3 dosyasını pygame ile yükleyip çalıyoruz
                pygame.mixer.music.play()  # Ses dosyasını çalmaya başlıyoruz
                while pygame.mixer.music.get_busy():  # Ses bitene kadar bekliyoruz
                    time.sleep(0.1)
        except Exception as e:
            print(f"Ses hatası: {e}")  # Eğer bir hata oluşursa hata mesajını yazdırıyoruz

    # Sesli yanıtı bir iş parçacığında (thread) başlatıyoruz
    threading.Thread(target=_speak, daemon=True).start()

# Görsel işleme fonksiyonu
def process_image(image_path):
    """Görüntü işleme fonksiyonu"""
    frame = cv2.imread(image_path)  # Yüklenen görseli okuyoruz
    if frame is None:
        return None, []  # Görsel okunamadıysa None döndürüyoruz

    results = model(frame)  # Modeli kullanarak görselde nesne tespiti yapıyoruz
    detected_objects = set()  # Algılanan nesneleri tutmak için bir küme

    # Modelin tespit ettiği nesneler üzerinde işlem yapıyoruz
    for result in results:
        for box in result.boxes:
            if box.conf[0] >= CONFIDENCE_THRESHOLD:  # Güven eşiği kontrolü
                class_name = model.names[int(box.cls[0])]  # Nesnenin sınıfını alıyoruz
                detected_objects.add(class_name)  # Algılanan nesneleri ekliyoruz

                # Nesnenin etrafına bounding box çiziyoruz
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Kutuyu çiziyoruz
                cv2.putText(frame, f"{class_name} {box.conf[0]:.2f}",
                            (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)  # Etiketi yazıyoruz

    # İşlenmiş görseli kaydediyoruz
    processed_filename = f"processed_{os.path.basename(image_path)}"
    processed_path = os.path.join(app.config['UPLOAD_FOLDER'], processed_filename)
    cv2.imwrite(processed_path, frame)  # İşlenmiş görseli kaydediyoruz

    return processed_path, list(detected_objects)  # İşlenmiş görselin yolunu ve tespit edilen nesneleri döndürüyoruz

# Video akışı oluşturma fonksiyonu
def generate_frames():
    """Video akışı oluşturma"""
    while cap.isOpened():  # Kamera açıldıkça döngü devam eder
        success, frame = cap.read()  # Bir kareyi alıyoruz
        if not success:
            break  # Eğer kare alınamazsa döngüyü bitiriyoruz

        frame = cv2.flip(frame, 1)  # Görüntüyü yatay olarak çeviriyoruz (aynaya benzer)
        current_time = time.time()  # Şu anki zamanı alıyoruz
        results = model(frame)  # Modelle nesne tespiti yapıyoruz
        current_detections = set()  # Geçerli tespit edilen nesneler kümesi

        # Nesneleri tespit edip üzerinde işlem yapıyoruz
        for result in results:
            for box in result.boxes:
                if box.conf[0] >= CONFIDENCE_THRESHOLD:  # Güven skoru kontrolü
                    class_name = model.names[int(box.cls[0])]  # Sınıf adı
                    current_detections.add(class_name)  # Nesneleri ekliyoruz

                    # Eğer bu nesne daha önce seslendirilmemişse, sesli bildirim yapıyoruz
                    if (class_name not in last_detected or
                            (current_time - last_detected[class_name]['time']) > MIN_SPEECH_INTERVAL):
                        speak(f"{class_name} algılandı")  # Sesli bildirim gönderiyoruz
                        last_detected[class_name] = {'time': current_time, 'alerted': True}  # Kayıtları güncelliyoruz

                    # Nesnenin etrafına bounding box çiziyoruz
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f"{class_name} {box.conf[0]:.2f}",
                                (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Video karesini JPEG formatında encode ediyoruz
        ret, buffer = cv2.imencode('.jpg', frame)
        if ret:
            # Akışa JPEG verisini ekliyoruz
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

# Ana sayfa route'u
@app.route('/')
def index():
    """Ana sayfa"""
    return render_template('index.html')  # index.html şablonunu render eder

# Video akışı için route
@app.route('/video_feed')
def video_feed():
    """Video akışı endpoint'i"""
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')  # Video akışı sağlanır

# Dosya yükleme route'u
@app.route('/upload', methods=['POST'])
def upload_file():
    """Dosya yükleme endpoint'i"""
    if 'file' not in request.files:
        flash('Dosya seçilmedi')  # Eğer dosya seçilmemişse hata mesajı göster
        return redirect(request.url)

    file = request.files['file']  # Yüklenen dosya alınır

    if file.filename == '':
        flash('Dosya seçilmedi')  # Dosya seçilmemişse hata mesajı göster
        return redirect(request.url)

    if not allowed_file(file.filename):
        flash('Geçersiz dosya türü')  # Geçersiz dosya türü durumunda hata mesajı göster
        return redirect(request.url)

    try:
        # Dosya ismi güvenli hale getirilir ve yüklenir
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Yüklenen görsel işlenir
        processed_path, detected_objects = process_image(filepath)
        if processed_path is None:
            flash('Görsel işlenemedi')  # Görsel işlenemezse hata mesajı göster
            return redirect(request.url)

        # İşlenmiş görsel ve tespit edilen nesnelerle sonuç sayfası render edilir
        return render_template('results.html',
                               original_image=f"uploads/{filename}",
                               processed_image=f"uploads/processed_{filename}",
                               objects=detected_objects)

    except Exception as e:
        flash(f'Hata oluştu: {str(e)}')  # Bir hata oluşursa mesaj göster
        return redirect(request.url)

# Uygulama başlatma
if __name__ == '__main__':
    try:
        # Flask uygulaması başlatılır
        app.run(host='0.0.0.0', port=5000, debug=True, threaded=True, use_reloader=False)
    finally:
        cap.release()  # Kamera kaynağını serbest bırakır
        pygame.mixer.quit()  # Pygame ses modülünü kapatır

