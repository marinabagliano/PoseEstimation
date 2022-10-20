# IA pose estimation with python/mediaPipe
# !pip install mediapipe opencv-python
     
# importando dependencias
import cv2 #importando openCv
import mediapipe as mp # importando mediaPipe (estimaçao de pose)
import numpy as np #trigonometria

# variaveis
mp_drawing = mp.solutions.drawing_utils # visualização de poses
mp_pose = mp.solutions.pose #modelo de estimação de pose

# funçoes
def calculate_angle(a,b,c):
    a = np.array(a) # First
    b = np.array(b) # Mid
    c = np.array(c) # End
    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0]) # conta matematica para calcular angulo entre 3 pontos
    angle = np.abs(radians*180.0/np.pi) # passando de radianos para graus
    
    if angle >180.0: # caso detectar o angulo da parte traseira do braço
        angle = 360-angle
        
    return angle # returna o angulo

cap = cv2.VideoCapture(0) # guardar captura a camera principal(0)

counter = 0 # contador de repetições
stage = None # estado(levantada pu abaixada)

## Configurar instâncias do mediapipe
#                 nivel minimo de detectação      nivel minimo de rastreamento
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    
    #enquanto a camera estiver ligada
    while cap.isOpened():
        ret, frame = cap.read()
        
        # Recolorir imagem para RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
      
        # Fazendo a detecção
        results = pose.process(image)
    
        # Recolorir novamente para BGR
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # Extraindo os marcadores
        try:
            #nomeando marcadores
            landmarks = results.pose_landmarks.landmark
            
            #Pegando as coordenadas
            #ombro/cotovelo/pulso
            shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
            wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
            
            # Calculando o angulo
            angle = calculate_angle(shoulder, elbow, wrist)
            
            
            # Visualizando o angulo
            # cv2.putText(imagem, texto, coord, fonte, fontScale, color,espessura, tipo de linha)
            cv2.putText(image, str(angle), 
                           tuple(np.multiply(elbow, [640, 480]).astype(int)), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                                )
            
            # logica para a contagem
            if angle > 160:
                stage = "down"
            if angle < 30 and stage =='down':
                stage="up"
                counter +=1
                       
        except:
            pass
        
        # Rendenizando as imagem
        # cv2.rectangle(imagem, (x1, y1), (x2, y2), cor, espesura)
        cv2.rectangle(image, (0,0), (225,73), (245,117,16), -1)
        
        # "printando" dados de reptição
        cv2.putText(image, 'REPS', (15,12), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
        cv2.putText(image, str(counter), 
                    (10,60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)
        
        # "printando" dados de estado
        cv2.putText(image, 'STAGE', (65,12), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
        cv2.putText(image, stage, 
                    (60,60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)
        
        
        # Colorindo as detecções
        #colorindo pose (linhas), cor, espesura
        #colorindo vertices (bolinha), cor, espesura, raio circulo
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), 
                                mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2) 
                                 )               
        
        #gerando "video"
        cv2.imshow('GYM', image)
        
        #finalizando programa
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
