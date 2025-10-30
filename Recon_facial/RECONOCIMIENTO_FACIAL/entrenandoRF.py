import cv2
import os
import numpy as np
from pathlib import Path

# Usar ruta relativa: carpeta DATA dentro de este directorio
base_dir = Path(__file__).resolve().parent
dataPath = base_dir / 'DATA'
if not dataPath.exists():
	print(f"No se encontró la carpeta DATA en {dataPath}. Crea {dataPath} con subcarpetas por persona y sus imágenes antes de entrenar.")
	raise SystemExit(1)

peopleList = os.listdir(dataPath)
print('Lista de personas: ', peopleList)

labels = []
facesData = []
label = 0

for nameDir in peopleList:
	personPath = str(dataPath / nameDir)
	print('Leyendo las imágenes')

	for fileName in os.listdir(personPath):
		print('Rostros: ', nameDir + '/' + fileName)
		labels.append(label)
		facesData.append(cv2.imread(personPath+'/'+fileName,0))
		#image = cv2.imread(personPath+'/'+fileName,0)
		#cv2.imshow('image',image)
		#cv2.waitKey(10)
	label = label + 1

#print('labels= ',labels)
#print('Número de etiquetas 0: ',np.count_nonzero(np.array(labels)==0))
#print('Número de etiquetas 1: ',np.count_nonzero(np.array(labels)==1))

# Métodos para entrenar el reconocedor
#face_recognizer = cv2.face.EigenFaceRecognizer_create()
#face_recognizer = cv2.face.FisherFaceRecognizer_create()
face_recognizer = cv2.face.LBPHFaceRecognizer_create()

# Entrenando el reconocedor de rostros
print("Entrenando...")
face_recognizer.train(facesData, np.array(labels))

# Almacenando el modelo obtenido
#face_recognizer.write('modeloEigenFace.xml')
#face_recognizer.write('modeloFisherFace.xml')
# Guardar el modelo en la ruta canónica en la raíz del proyecto (dos niveles arriba: Recon_facial/)
canonical_path = Path(__file__).resolve().parents[1] / 'modeloLBPHFace.xml'
face_recognizer.write(str(canonical_path))
print(f"Modelo almacenado en: {canonical_path}")