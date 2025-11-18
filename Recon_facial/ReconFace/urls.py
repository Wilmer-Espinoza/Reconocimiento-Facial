from django.urls import path
from . import views

urlpatterns = [
    path('', views.reconocimiento, name='reconocimiento'),                             # página principal
    path('reconocimiento/', views.reconocimiento, name='reconocimiento'),
    path('video_feed/', views.video_feed, name='video_feed'),      # stream MJPEG
    # no exponemos endpoints de captura/entrenamiento por UI (solicitado)
    path('model_status/', views.model_status, name='model_status'),  # indica si hay un modelo disponible
    path('camera_status/', views.camera_status, name='camera_status'),
    path('snapshot/', views.snapshot, name='snapshot'),
    path('current_detections/', views.current_detections, name='current_detections'),
    path('attendance_today/', views.attendance_today, name='attendance_today'),
]