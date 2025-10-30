from django.shortcuts import render, redirect
from django.http import StreamingHttpResponse, JsonResponse, HttpResponse
from django.views.decorators.http import require_http_methods
from django.conf import settings
import cv2, os, time
import threading
import numpy as np
import shutil
import json
from datetime import datetime, timezone

# Helper: creación robusta del reconocedor LBPH (varias APIs según la build de OpenCV)
def create_lbp_recognizer():
    """Intentar crear un recognizer LBPH usando distintas firmas según la versión/build.
    Devuelve el objeto recognizer o None si no está disponible (p.ej. falta opencv-contrib).
    """
    try:
        # API moderna (opencv-contrib)
        if hasattr(cv2, 'face') and hasattr(cv2.face, 'LBPHFaceRecognizer_create'):
            return cv2.face.LBPHFaceRecognizer_create()
    except Exception:
        pass
    try:
        # API alternativa antigua
        if hasattr(cv2, 'face') and hasattr(cv2.face, 'createLBPHFaceRecognizer'):
            return cv2.face.createLBPHFaceRecognizer()
    except Exception:
        pass
    try:
        # otra variante en algunas builds
        if hasattr(cv2, 'createLBPHFaceRecognizer'):
            return cv2.createLBPHFaceRecognizer()
    except Exception:
        pass
    return None

# === Rutas utilitarias ===
HAAR_PATH = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
# DATA_DIR: preferimos la carpeta existente (underscore) usada por los scripts originales
possible1 = os.path.join(settings.BASE_DIR, 'RECONOCIMIENTO_FACIAL', 'DATA')
possible2 = os.path.join(settings.BASE_DIR, 'RECONOCIMIENTO FACIAL', 'DATA')
if os.path.exists(possible1):
    DATA_DIR = possible1
else:
    DATA_DIR = possible2
# Rutas de modelo: preferimos el modelo en la raíz del proyecto (pract) y mantenemos
# una copia en MEDIA para compatibilidad con la app web.
MODEL_DIR = os.path.join(settings.MEDIA_ROOT, 'models')
os.makedirs(MODEL_DIR, exist_ok=True)
MEDIA_MODEL_PATH = os.path.join(MODEL_DIR, 'modeloLBPHFace.xml')   # copia para MEDIA
ROOT_MODEL_PATH = r'C:\Users\usuario\Desktop\pract\modeloLBPHFace.xml'
CASCADE = cv2.CascadeClassifier(HAAR_PATH)

# Umbral de confianza para LBPH (menor = mejor). Usamos la misma lógica que ReconocimientoFacial.py
LBPH_THRESHOLD = 70.0

# Directorio para guardar asistencias por día (JSON)
ATT_DIR = os.path.join(settings.BASE_DIR, 'attendance')
os.makedirs(ATT_DIR, exist_ok=True)

# Sincronizar ambas ubicaciones: si hay modelo en la raíz, copiar/reescribir en MEDIA.
# Si no hay en root pero existe en MEDIA, copiar a la raíz para asegurar que la
# ruta raíz siempre esté disponible como fuente preferente.
try:
    if os.path.exists(ROOT_MODEL_PATH):
        try:
            shutil.copy2(ROOT_MODEL_PATH, MEDIA_MODEL_PATH)
        except Exception:
            # no crítico; la app seguirá funcionando si solo una copia está disponible
            pass
    elif os.path.exists(MEDIA_MODEL_PATH) and not os.path.exists(ROOT_MODEL_PATH):
        try:
            os.makedirs(os.path.dirname(ROOT_MODEL_PATH), exist_ok=True)
            shutil.copy2(MEDIA_MODEL_PATH, ROOT_MODEL_PATH)
        except Exception:
            pass
except Exception:
    pass

# ===== Cámara singleton =====
class Camera:
    def __init__(self, src=0):
        """Inicializa la cámara usando DirectShow en Windows y prepara estado.
        Se crea un lock para accesos concurrentes y una lista con las últimas detecciones.
        """
        self.cap = None
        self.lbp = None
        self.capturing = False
        self.person_name = "Wilmer"
        # Lock para proteger accesos concurrentes a la cámara
        self.lock = threading.Lock()
        # Últimas detecciones: lista de dicts {name, conf}
        self.last_detections = []
        # Cache para evitar escrituras excesivas en disco: {name: last_timestamp}
        self.attendance_cache = {}
        self.attendance_lock = threading.Lock()

        print("Iniciando cámara con DirectShow...")
        try:
            # Forzar uso de DirectShow y configurar propiedades
            self.cap = cv2.VideoCapture(src, cv2.CAP_DSHOW)

            # Configurar propiedades de la cámara
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            try:
                self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
            except Exception:
                pass

            # Verificar si la cámara se abrió
            if not self.cap.isOpened():
                print("Error: La cámara no se pudo abrir con DirectShow")
                # Intento de respaldo con backend por defecto
                self.cap = cv2.VideoCapture(src)
                if not self.cap.isOpened():
                    print("Error: Tampoco funciona el backend por defecto")
            else:
                # Verificar que podemos leer frames
                ret, test_frame = self.cap.read()
                if not ret or test_frame is None:
                    print("Error: No se pueden leer frames de la cámara")

            if self.cap is not None and self.cap.isOpened():
                print(f"Cámara iniciada exitosamente:")
                try:
                    print(f"- Resolución: {int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")
                    print(f"- FPS: {int(self.cap.get(cv2.CAP_PROP_FPS))}")
                except Exception:
                    pass

        except Exception as e:
            print(f"Error inicializando cámara: {str(e)}")
            if self.cap:
                try:
                    self.cap.release()
                except Exception:
                    pass
                self.cap = None

        # Nota: si no se pudo abrir, intentar con backend por defecto (último recurso)
        if self.cap is None or not self.cap.isOpened():
            print("¡ADVERTENCIA! No se pudo inicializar ninguna cámara")
            try:
                self.cap = cv2.VideoCapture(0)
            except Exception:
                self.cap = None

        self.load_model()

    def load_model(self):
        # Preferir el modelo ubicado en la raíz del proyecto; si no existe, usar la copia en MEDIA
        model_to_load = None
        if os.path.exists(ROOT_MODEL_PATH):
            model_to_load = ROOT_MODEL_PATH
        elif os.path.exists(MEDIA_MODEL_PATH):
            model_to_load = MEDIA_MODEL_PATH

        if model_to_load:
            try:
                self.lbp = create_lbp_recognizer()
                if self.lbp is None:
                    # recognizer no disponible en esta build de OpenCV
                    self.lbp = None
                    return
                self.lbp.read(model_to_load)
            except Exception:
                self.lbp = None

    def record_attendance(self, name, conf):
        """Registra en un archivo JSON por fecha la asistencia de 'name'.
        Guarda first_seen, last_seen y best_conf (mejor/conf mínimo).
        """
        date_str = datetime.now().strftime('%Y-%m-%d')
        path = os.path.join(ATT_DIR, f"{date_str}.json")
        with self.attendance_lock:
            try:
                if os.path.exists(path):
                    with open(path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                else:
                    data = {}

                now_iso = datetime.now(timezone.utc).isoformat()
                entry = data.get(name, None)
                if not entry:
                    entry = {
                        'name': name,
                        'first_seen': now_iso,
                        'last_seen': now_iso,
                        'best_conf': conf
                    }
                else:
                    entry['last_seen'] = now_iso
                    # LBPH: lower confidence = better
                    try:
                        entry['best_conf'] = min(entry.get('best_conf', conf), conf)
                    except Exception:
                        entry['best_conf'] = conf

                data[name] = entry

                tmp = path + '.tmp'
                with open(tmp, 'w', encoding='utf-8') as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
                os.replace(tmp, path)
            except Exception as e:
                print(f"Error recording attendance: {e}")

    def predict_label(self, face_img_gray):
        if self.lbp is None:
            return ("[Modelo no cargado]", 0)
        try:
            label, conf = self.lbp.predict(face_img_gray)
            # Mapea label->nombre usando carpetas de DATA (ordenadas)
            names = sorted([d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))])
            name = names[label] if 0 <= label < len(names) else f"ID {label}"
            return (f"{name}", conf)
        except Exception:
            return ("Desconocido", 0)

    def get_frame(self):
        """Obtiene un frame de la cámara con reintentos y procesamiento de rostros."""
        if self.cap is None:
            print("Error: Objeto de cámara no existe")
            return None
            
        if not self.cap.isOpened():
            print("Error: Cámara cerrada, intentando reabrir...")
            try:
                self.cap.release()
                self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
                if not self.cap.isOpened():
                    print("Error: No se pudo reabrir la cámara")
                    return None
            except Exception as e:
                print(f"Error reabriendo cámara: {e}")
                return None

        # Intentar leer frame con reintentos
        frame = None
        max_retries = 3
        for attempt in range(max_retries):
            try:
                with self.lock:
                    ok, frame = self.cap.read()
                if ok and frame is not None:
                    break
                print(f"Intento {attempt + 1}/{max_retries}: No se pudo leer frame")
                time.sleep(0.1)
            except Exception as e:
                print(f"Error leyendo frame (intento {attempt + 1}): {e}")
                time.sleep(0.1)
        
        if frame is None:
            print("Error: No se pudo obtener frame después de reintentos")
            return None
        try:
            # Verificar frame válido
            if frame.size == 0 or frame.shape[0] == 0 or frame.shape[1] == 0:
                print("Error: Frame vacío o inválido")
                return None

            # Convertir a escala de grises y detectar rostros
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = CASCADE.detectMultiScale(gray, 1.3, 5)
            
            # Procesar cada rostro detectado
            detections = []
            for (x,y,w,h) in faces:
                if x < 0 or y < 0 or x+w > frame.shape[1] or y+h > frame.shape[0]:
                    continue  # Ignorar rostros con coordenadas inválidas
                
                try:
                    # Extraer y procesar rostro
                    face = gray[y:y+h, x:x+w]
                    # Usar el mismo preprocesado que ReconocimientoFacial.py (150x150)
                    face_resized = cv2.resize(face, (150,150))
                    name, conf = self.predict_label(face_resized)
                    # Si la confianza es alta (valor grande), consideramos 'Desconocido'
                    is_recognized = (conf < LBPH_THRESHOLD and name not in ("[Modelo no cargado]","Desconocido"))
                    text = name if is_recognized else "Desconocido"
                    
                    # Dibujar rectángulo y texto
                    cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)
                    cv2.putText(frame, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
                    
                    # Guardar rostro si estamos capturando
                    if self.capturing:
                        person_dir = os.path.join(DATA_DIR, self.person_name)
                        os.makedirs(person_dir, exist_ok=True)
                        fname = f"rostro_{int(time.time()*1000)}.jpg"
                        cv2.imwrite(os.path.join(person_dir, fname), face_resized)
                    # Añadir detección (solo si reconocido)
                    try:
                        if is_recognized:
                            detections.append({"name": name, "conf": float(conf)})
                        else:
                            # opcional: registrar 'Desconocido' si quieres
                            pass
                    except Exception:
                        detections.append({"name": str(name), "conf": 0.0})

                    # Registrar asistencia (no bloqueante intensivo)
                    try:
                        # Solo registrar asistencia si fue reconocido
                        if is_recognized:
                            now_ts = time.time()
                            last_ts = self.attendance_cache.get(name, 0)
                            if now_ts - last_ts > 30:
                                try:
                                    self.record_attendance(name, float(conf))
                                    self.attendance_cache[name] = now_ts
                                except Exception:
                                    pass
                    except Exception:
                        pass
                except Exception as e:
                    print(f"Error procesando rostro: {e}")
                    continue  # Continuar con el siguiente rostro
            
            # Guardar últimas detecciones en el objeto cámara
            try:
                with self.lock:
                    self.last_detections = detections[:10]
            except Exception:
                self.last_detections = detections[:10]

            # Codificar frame a JPEG
            try:
                ret, jpeg = cv2.imencode('.jpg', frame)
                if not ret:
                    print("Error: No se pudo codificar frame a JPEG")
                    return None
                return jpeg.tobytes()
            except Exception as e:
                print(f"Error codificando frame a JPEG: {e}")
                return None
        except Exception as e:
            print(f"Error procesando frame: {e}")
            return None
                
    def release(self):
        if self.cap:
            self.cap.release()

cam = None
def get_cam():
    global cam
    if cam is None:
        # Cambia a 0 para webcam; o a ruta de tu 'Video.mp4'
        cam = Camera(src=0)  
    return cam

# ===== Generador para streaming MJPEG =====
def gen(camera):
    frame_count = 0
    error_count = 0
    MAX_ERRORS = 3  # máximo de errores consecutivos antes de terminar

    while True:
        try:
            frame = camera.get_frame()
            if frame is None:
                error_count += 1
                print(f"Error obteniendo frame ({error_count}/{MAX_ERRORS})")
                if error_count >= MAX_ERRORS:
                    print("Demasiados errores consecutivos, cerrando stream")
                    break
                time.sleep(0.1)  # breve pausa antes de reintentar
                continue
            
            # Reset contador de errores si obtuvimos un frame
            error_count = 0
            frame_count += 1
            if frame_count % 30 == 0:  # log cada 30 frames
                print(f"Stream activo: {frame_count} frames enviados")
            
            # Enviar frame como parte del stream MJPEG
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
                   
        except Exception as e:
            print(f"Error en generador de stream: {e}")
            error_count += 1
            if error_count >= MAX_ERRORS:
                print("Demasiados errores, cerrando stream")
                break
            time.sleep(0.1)  # pausa antes de reintentar

# ===== Vistas =====
def home(request):
    return render(request, 'base_template.html')  # o crea un index propio si prefieres

def reconocimiento(request):
    return render(request, 'reconocimiento/index.html')  # lo creamos en el paso 4

def video_feed(request):
    camera = get_cam()
    # Si la cámara no pudo abrirse, devolvemos un mensaje sencillo (el cliente JS mostrará aviso)
    try:
        cap_ok = camera.cap.isOpened()
    except Exception:
        cap_ok = False
    if not cap_ok:
        return JsonResponse({"ok": False, "error": "No se puede abrir la cámara en el servidor."}, status=503)
    return StreamingHttpResponse(gen(camera), content_type='multipart/x-mixed-replace; boundary=frame')


def model_status(request):
    """
    Devuelve si hay un modelo LBPH disponible en el servidor.
    Esto permite a la UI deshabilitar el botón de "Reconocer" si no existe el modelo.
    """
    try:
        exists_root = os.path.exists(ROOT_MODEL_PATH)
        exists_media = os.path.exists(MEDIA_MODEL_PATH)
        exists = exists_root or exists_media
    except Exception:
        exists = False
        exists_root = False
        exists_media = False

    model_path = ROOT_MODEL_PATH if exists_root else (MEDIA_MODEL_PATH if exists_media else "")
    return JsonResponse({"model_exists": exists, "model_path": model_path})


def camera_status(request):
    """Comprobación rápida de la webcam en el servidor. No inicializa el singleton compartido.
    Abre y cierra un VideoCapture temporal para no interferir con la cámara en uso.
    """
    ok = False
    backends = []
    try:
        # intento rápido con backends recomendados en Windows
        for backend in (None, cv2.CAP_DSHOW, cv2.CAP_MSMF):
            try:
                if backend is None:
                    cap = cv2.VideoCapture(0)
                else:
                    cap = cv2.VideoCapture(0, backend)
                opened = cap.isOpened()
                backends.append({"backend": backend if backend is not None else "default", "opened": bool(opened)})
                if opened:
                    ok = True
                    cap.release()
                    break
                cap.release()
            except Exception as e:
                backends.append({"backend": backend if backend is not None else "default", "error": str(e)})
    except Exception:
        ok = False
    return JsonResponse({"camera_ok": ok, "tried": backends})


def snapshot(request):
    """Devuelve un único frame JPEG para diagnóstico rápido en el frontend."""
    try:
        camera = get_cam()
        # En lugar de leer camera.cap directamente (lo que genera condiciones de carrera
        # con el stream), reutilizamos get_frame() que ya implementa lock, reintentos
        # y almacenamiento en camera.last_detections. Llamamos a get_frame() para forzar
        # una captura/procesado y luego devolvemos las detecciones almacenadas.
        try:
            jpg = camera.get_frame()
        except Exception as e:
            return JsonResponse({"ok": False, "error": f"Error al capturar frame: {e}"}, status=500)

        if jpg is None:
            # No pudo obtenerse frame/imagen -> servicio no disponible para snapshot
            return JsonResponse({"ok": False, "error": "No frame available"}, status=503)

        # Leer las detecciones calculadas por get_frame()
        try:
            with camera.lock:
                detections = list(getattr(camera, 'last_detections', []))
        except Exception:
            detections = getattr(camera, 'last_detections', [])

        return JsonResponse({"ok": True, "faces": detections})
    except Exception as e:
        return JsonResponse({"ok": False, "error": str(e)}, status=500)


def current_detections(request):
    """Devuelve las detecciones más recientes (sin recapturar la cámara).
    Útil para que el frontend muestre en tiempo real los nombres detectados.
    """
    try:
        camera = get_cam()
        # Leer detecciones de forma thread-safe
        try:
            with camera.lock:
                det = list(camera.last_detections)
        except Exception:
            det = getattr(camera, 'last_detections', [])
        return JsonResponse({"faces": det})
    except Exception as e:
        return JsonResponse({"faces": [], "error": str(e)}, status=500)


def attendance_today(request):
    """Devuelve la lista de asistencias registradas para la fecha actual.
    Lee el JSON en disk guardado por Camera.record_attendance().
    """
    try:
        date_str = datetime.now().strftime('%Y-%m-%d')
        path = os.path.join(ATT_DIR, f"{date_str}.json")
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            items = list(data.values())
            # ordenar por first_seen
            items.sort(key=lambda x: x.get('first_seen', ''))
            return JsonResponse({"attendance": items})
        else:
            return JsonResponse({"attendance": []})
    except Exception as e:
        return JsonResponse({"attendance": [], "error": str(e)}, status=500)

@require_http_methods(["POST"])
def toggle_capture(request):
    """
    Enciende/apaga la recolección de recortes para el dataset.
    Optional: recibe 'person_name'
    """
    camera = get_cam()
    pn = request.POST.get('person_name')
    if pn:
        camera.person_name = pn.strip() or camera.person_name
    camera.capturing = not camera.capturing
    return JsonResponse({"capturing": camera.capturing, "person_name": camera.person_name})

@require_http_methods(["POST"])
def train_model(request):
    """
    Re-entrena LBPH con las carpetas en DATA/
    """
    # requiere opencv-contrib-python
    images, labels = [], []
    names = sorted([d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))])
    if not names:
        return JsonResponse({"ok": False, "error": "No hay carpetas de personas en DATA/"}, status=400)

    for label, name in enumerate(names):
        pdir = os.path.join(DATA_DIR, name)
        for fname in os.listdir(pdir):
            if fname.lower().endswith(('.jpg','.png','.jpeg')):
                img = cv2.imread(os.path.join(pdir, fname), cv2.IMREAD_GRAYSCALE)
                if img is None: 
                    continue
                img = cv2.resize(img, (200,200))
                images.append(img)
                labels.append(label)

    if not images:
        return JsonResponse({"ok": False, "error": "No se encontraron imágenes válidas"}, status=400)

    # crear recognizer de forma robusta
    lbp = create_lbp_recognizer()
    if lbp is None:
        return JsonResponse({"ok": False, "error": "LBPH recognizer no disponible. Instala opencv-contrib-python."}, status=500)
    lbp.train(images, np.array(labels))
    # Guardar modelo tanto en la raíz del proyecto como en MEDIA para consistencia
    try:
        os.makedirs(os.path.dirname(ROOT_MODEL_PATH), exist_ok=True)
        lbp.write(ROOT_MODEL_PATH)
    except Exception:
        pass
    try:
        os.makedirs(MODEL_DIR, exist_ok=True)
        lbp.write(MEDIA_MODEL_PATH)
    except Exception:
        pass

    # recarga en la cámara viva
    get_cam().load_model()
    # devolver la ruta preferente (raíz si existe)
    preferred = ROOT_MODEL_PATH if os.path.exists(ROOT_MODEL_PATH) else (MEDIA_MODEL_PATH if os.path.exists(MEDIA_MODEL_PATH) else "")
    return JsonResponse({"ok": True, "person_folders": names, "model_path": preferred})