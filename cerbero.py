import cv2
from statistics import mean
import numpy as np

CONFIDENCE_THRESHOLD = 0.2
NMS_THRESHOLD = 0.4

listCharacters = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N',
                  'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '0', '1',
                  '2', '3', '4', '5', '6', '7', '8', '9']

netVehicle = cv2.dnn.readNet("pesos/veiculo.weights", "configuracoes/veiculo.cfg")
netVehicle.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
netVehicle.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA_FP16)
modelVehicle = cv2.dnn_DetectionModel(netVehicle)
modelVehicle.setInputParams(size=(416, 416), scale=1 / 255)

netPlate = cv2.dnn.readNet("pesos/placa.weights", "configuracoes/placa.cfg")
netPlate.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
netPlate.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA_FP16)
modelPlate = cv2.dnn_DetectionModel(netPlate)
modelPlate.setInputParams(size=(416, 416), scale=1 / 255)

netRegion = cv2.dnn.readNet("pesos/regiao.weights", "configuracoes/regiao.cfg")
netRegion.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
netRegion.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA_FP16)
modelRegion = cv2.dnn_DetectionModel(netRegion)
modelRegion.setInputParams(size=(416, 416), scale=1 / 255)

netCharacter = cv2.dnn.readNet("pesos/caractere.weights", "configuracoes/caractere.cfg")
netCharacter.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
netCharacter.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA_FP16)
modelCharacter = cv2.dnn_DetectionModel(netCharacter)
modelCharacter.setInputParams(size=(416, 416), scale=1 / 255)

# Usando video
cap = cv2.VideoCapture('demonstracao.mp4')

# Usando camera, aletar para http://usuario:senha@ip:porta/video
# cap = cv2.VideoCapture('http://admin:123456@192.168.2.111:8090/video')

count = 0

while cap.isOpened():
    ret, frame = cap.read()
    classesPlate, scoresPlate, boxesPlate = modelPlate.detect(frame, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)
    cap.set(cv2.CAP_PROP_POS_MSEC, (count * 1000))

    for d in boxesPlate:
        classesVehicle, scoresVehicle, boxesVehicle = modelVehicle.detect(frame, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)
        for c in boxesVehicle:
            xVehicle, yVehicle, wVehicle, hVehicle = c
            vehicle = frame[yVehicle:yVehicle + hVehicle, xVehicle:xVehicle + wVehicle]

        try:
            xPlate, yPlate, wPlate, hPlate = d
            plate = frame[yPlate:yPlate + hPlate, xPlate:xPlate + wPlate]
            height, _ = plate.shape[:2]
            x = 318 / height
            plate = cv2.resize(plate, (0, 0), fx=x, fy=x, interpolation=cv2.INTER_AREA)
            height, _, _ = np.shape(plate)
            avg_color_per_row = np.average(plate, axis=0)
            avg_colors = np.average(avg_color_per_row, axis=0)
            int_averages = np.array(avg_colors, dtype=np.uint8)
            b = int_averages[0]
            g = int_averages[1]
            r = int_averages[2]
            if 0 < b < 85 and 0 < g < 85 and 130 < r < 255:
                plate = plate - 255

            blurPlate = cv2.pyrMeanShiftFiltering(plate, 10, 40)
            grayPlate = cv2.cvtColor(blurPlate, cv2.COLOR_BGR2GRAY)
            grayPlate = cv2.medianBlur(grayPlate, 11)
            grayPlate = cv2.cvtColor(grayPlate, cv2.COLOR_GRAY2BGR)

            classesRegion, scoresRegion, boxesRegion = modelRegion.detect(grayPlate, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)
            contours = []
            modeRegion = []
            for e in boxesRegion:
                modeRegion.append(e[3])
            modeRegionOrdered = sorted(modeRegion)
            del (modeRegionOrdered[0])
            del (modeRegionOrdered[len(modeRegionOrdered) - 1])
            meanRegion = mean(modeRegionOrdered)

            for e in boxesRegion:
                if meanRegion + meanRegion * 0.3 > e[3] > meanRegion - meanRegion * 0.3:
                    contours.append(e)
                    contours = sorted(contours, key=lambda cont: cont[0])
                    ocrPlate = ''
                for f in contours:
                    xRegion, yRegion, wRegion, hRegion = f
                    character = grayPlate[yRegion:yRegion + hRegion, xRegion:xRegion + wRegion]
                    thresh = cv2.threshold(character, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
                    invert = 255 - thresh
                    invert = cv2.cvtColor(invert, cv2.COLOR_GRAY2BGR)
                    height, width, channels = character.shape
                    whiteImage = np.zeros((height * 2, width * 2, 3), dtype=np.uint8)
                    cv2.rectangle(whiteImage, (0, 0), (width * 2, height * 2), (255, 255, 255), -1)
                    x_offset = int(width * 0.25)
                    y_offset = int(height * 0.25)
                    whiteImage[y_offset:y_offset + invert.shape[0], x_offset:x_offset + invert.shape[1]] = invert
                    x_offset = int(width * 0.25)
                    y_offset = int(height * 0.25)
                    whiteImage[y_offset:y_offset + invert.shape[0], x_offset:x_offset + invert.shape[1]] = invert
                    classesCharacter, scoresCharacter, boxesCharacter = modelCharacter.detect(whiteImage, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)
                    ocrPlate += listCharacters[int(classesCharacter[0])]
            parte1 = ocrPlate[:3]
            parte2 = ocrPlate[3]
            parte3 = ocrPlate[4]
            parte4 = ocrPlate[5:]

            parte1 = parte1.replace('1', 'I')
            parte1 = parte1.replace('0', 'O')

            parte2 = parte2.replace('I', '1')
            parte2 = parte2.replace('O', '0')

            parte3 = parte3.replace('I', '1')
            parte3 = parte3.replace('O', '0')

            parte4 = parte4.replace('I', '1')
            parte4 = parte4.replace('O', '0')
            ocrPlate = parte1 + parte2 + parte3 + parte4
            print(f'{ocrPlate}')
        except:
            None
    count += 1
