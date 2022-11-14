# import the necessary packages
from __future__ import print_function
# from basicmotiondetector import BasicMotionDetector
from imutils.video import VideoStream
import numpy as np
import datetime
import imutils
import time
import cv2
import math
from collections import Counter
import xml.etree.ElementTree as ET
tree = ET.parse('settings.xml')
root = tree.getroot()



import socket

HOST = '192.168.0.10'  # Standard loopback interface address (localhost)


# HOST = '192.168.0.245'  # 
# HOST = '192.168.1.102'
PORT = 8001        # Port to listen on (non-privileged ports are > 1023)
import json

import threading
import time

import random

pomerajX=103.43
pomerajY=20.08

def json_create(Cx,Cy,Cz,Alpha,Rotx,Rotz,Tip,Boja):
    b={}
    b['Cx']=Cx
    b['Cy']=Cy
    b['Cz']=Cz
    b['Alpha']=Alpha
    b['Rotx']=Rotx
    b['Rotz']=Rotz
    b['Tip']=Tip
    b['B']=Boja
    return b

def comm():
    
    global tela
    
#     host = socket.gethostname()
#     
#     port = 8001  # initiate port no above 1024
# 
#     server_socket = socket.socket()  # get instance
#     # look closely. The bind() function takes tuple as argument
#     server_socket.bind((HOST, PORT))  # bind host address and port together
# 
#     # configure how many client the server can listen simultaneously
#     server_socket.listen(1)
#     conn, address = server_socket.accept()  # accept new connection
#     print("Connection from: " + str(address))
#     while True:
#         # receive data stream. it won't accept data packet greater than 1024 bytes
#         data = conn.recv(1024)
#         if not data:
#             # if data is not received break
#             break
#         print("from connected user: " + str(data))
#         data = input(' -> ')
#         conn.sendall(tela.encode())  # send data to the client
# 
#     conn.close()    
    
   
   
#     s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
#     s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
#     s.bind((HOST, PORT))
#     s.listen(10)
#     conn, addr = s.accept()
# 
#     with conn:
#         print('Connected by', addr)
#         while True:
#             data = conn.recv(1024)
#             if not data:
#                 print('close')
# #                 break
#             conn.sendall(tela.encode())
#            
#     conn.close()
#     print('close')
#     time.sleep(0.5)
    for k in (1,10000):
       
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
           
            s.bind((HOST, PORT))
            s.listen()
            
            conn, addr = s.accept()
            
            with conn:
                print('Connected by', addr)
                while True:
                    try:
                        data = conn.recv(1024)
                    except:
                        data=None
                    if not data:
                        
                        conn.close()
                        conn, addr = s.accept()
                        
                        
                    conn.sendall(tela.encode())
                    print(tela)
                    conn.close()  #dodato
            
    #             print(repr(data)) 

geom_tela = [{"dimenzije": [25.,25.,7.],"model": "A",},{"dimenzije": [25.,50.,7.],"model": "B",},{"dimenzije": [11.,50.,7.],"model": "C",}]
def real2ref(real,ref,tol):

    if ((ref+ref*tol/100.>real)and (ref-ref*tol/100.<real)):

        return True

    else:

        return False

AA=[25.,25.,8.]
BB=[25.,50.,8.]
CC=[17.,50.,8.]
countA = Counter(AA)
countB = Counter(BB)
countC = Counter(CC)
All=[17,25.,50.,8.]


# def treca(a,b,tip):
#     
#     
#     if tip=='A':
#         skup=[25.,25.,8.]
#     elif  tip=='B':
#         skup=[25.,50.,8.]
#     elif  tip=='C':
#         skup=[17.,50.,8.]
# 
#     skup.remove(a)
#     skup.remove(b)
#     return skup[0]
    
def u_toleranciji(aprox,tol):
    for i in All:
        if ((i+i*tol/100.>aprox)and (i-i*tol/100.<aprox)):
            return i
    return None
def pretrazivanje(x_real,y_real,tol):
    x_aprox=u_toleranciji(x_real,tol)
    y_aprox=u_toleranciji(y_real,tol)
   
    if x_aprox!=None and y_aprox!=None:
        D=Counter([x_aprox,y_aprox])
        
        if  sum((countA & D).values())==2:
            if (x_aprox!=8. and y_aprox!=8. ):
                return [x_aprox,y_aprox,8./2,0,0,'A']
                
            else:
                return [x_aprox,y_aprox,25./2,90,0,'A']
        elif sum((countB & D).values())==2:   
            if (x_aprox!=8. and y_aprox!=8.):    
                return [x_aprox,y_aprox,8./2,0,0,'B']
            elif (x_aprox!=25. and y_aprox!=25.):
                return [x_aprox,y_aprox,25./2,0,90,'B']
            else:
                return [x_aprox,y_aprox,50./2,90,0,'B']
        elif sum((countC & D).values())==2:
            if (x_aprox!=8. and y_aprox!=8.):    
                return [x_aprox,y_aprox,8./2,0,0,'C']
            elif (x_aprox!=50. and y_aprox!=50.):
                return [x_aprox,y_aprox,50./2,90,0,'C']
            else:
                return [x_aprox,y_aprox,17./2,0,90,'C']
            
            
                
    return None

def nothing(x):
    pass

def distanca(ax,ay,bx,by):
    return math.sqrt(pow(ax-bx,2)+pow(ay-by,2))
# initialize the video streams and allow them to warmup
def kamere():
    global tela
    niz1=[{"Cx": 139.22, "Cy": 123, "Cz": 4.0, "Alpha": 102.44, "Rotx": 0, "Rotz": 0, "Tip": "A"}]
    print("[INFO] starting cameras...")

#     webcam = VideoStream(src=0,framerate=10).start()
    picam = VideoStream(usePiCamera=True,resolution=(1920, 1088),framerate=10).start()
    time.sleep(1.0)
    print("[INFO] starting cameras2...")
    # initialize the two motion detectors, along with the total
    # number of frames read
    # camMotion = BasicMotionDetector()
    # piMotion = BasicMotionDetector()
    total = 0
    pomeraj_x=0
    pomeraj_y=640
    parametri={}
    boje=['plava','crvena','zuta','zelena']
    for elem2 in root.findall('skaliranje'):
        for subelem2 in elem2:
            nn=subelem2.get('name')
            if nn == 'odnos':
                odnos=float(subelem2.text)
#     print(odnos)            
    for i in boje:
        for elem in root.findall(i):
            for subelem in elem:
                nn=subelem.get('name')
                parametri[i+'_'+nn]=int(subelem.text)
    #             if nn == 'h1':
    #                 
    #                 h1=int(subelem.text)
    #             elif nn == 'h2':
    #                 h2=int(subelem.text)
    #             elif nn == 's1':
    #                 s1=int(subelem.text)
    #             elif nn == 's2':
    #                 s2=int(subelem.text)
    #             elif nn == 'v1':
    #                 v1=int(subelem.text)
    #             elif nn == 'v2':
    #                 v2=int(subelem.text)
    #                 print(v2)
    #


                     
    # loop over frames from the video streams
    ok=True
    time.sleep(1.0)
    while ok:
        a=[]
        
        
        real_time_object=[]  
        # initialize the list of frames that have been processed
        frame1 = []
#         frame1 = webcam.read()
        frame2 = picam.read()
        
        frame2 = frame2[pomeraj_x:pomeraj_x+480,pomeraj_y:pomeraj_y+640]
        frame2=cv2.flip(frame2, 1)
#         frame2=cv2.rotate(frame2, cv2.ROTATE_180)
        
        hsv_top=cv2.cvtColor(frame2,cv2.COLOR_BGR2HSV)
       
        b1=np.array([parametri['crvena_h1'],parametri['crvena_s1'],parametri['crvena_v1']])
        b2=np.array([parametri['crvena_h2'],parametri['crvena_s2'],parametri['crvena_v2']])
        mask_crvena=cv2.inRange(hsv_top,b1,b2)
        b1=np.array([parametri['plava_h1'],parametri['plava_s1'],parametri['plava_v1']])
        b2=np.array([parametri['plava_h2'],parametri['plava_s2'],parametri['plava_v2']])
        mask_plava=cv2.inRange(hsv_top,b1,b2)
        b1=np.array([parametri['zuta_h1'],parametri['zuta_s1'],parametri['zuta_v1']])
        b2=np.array([parametri['zuta_h2'],parametri['zuta_s2'],parametri['zuta_v2']])
        mask_zuta=cv2.inRange(hsv_top,b1,b2)
        b1=np.array([parametri['zelena_h1'],parametri['zelena_s1'],parametri['zelena_v1']])
        b2=np.array([parametri['zelena_h2'],parametri['zelena_s2'],parametri['zelena_v2']])
        mask_zelena=cv2.inRange(hsv_top,b1,b2)
        
        
    #     res_top_crvena=cv2.bitwise_or(frame2,frame2,mask=mask_crvena)
    #     res_top_plava=cv2.bitwise_and(frame2,frame2,mask=mask_plava)
    #     res_top_zuta=cv2.bitwise_and(frame2,frame2,mask=mask_zuta)
    #     res_top_zelena=cv2.bitwise_and(frame2,frame2,mask=mask_zelena)
        res_top1=cv2.bitwise_or(mask_crvena,mask_plava)
        res_top2=cv2.bitwise_or(mask_zuta,mask_zelena)
        mask=cv2.bitwise_or(res_top1,res_top2)
    #     res_top=cv2.bitwise_not(mask)
    #     res_top=cv2.Canny(res_top,100,100)
        
    #     gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    #     thresh = cv2.threshold(res_top, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    #     print(thresh)
    #     cv2.imshow('thresh', thresh)
        cnts = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts=imutils.grab_contours(cnts)
        
    #     print(len(cnts[1]))
    #     cnts = cnts[0] if len(cnts) == 2 else cnts[1]
        
        for cnt in cnts:
    #         if cv2.contourArea(cnt)<50:
    #             continue
            
    #         try:    
            rect = cv2.minAreaRect(cnt)
    #                 M=cv2.moments(cnt)
#             M=cv2.moments(cnt)
#             #teziste cetvorougla
#             Cx=int(M['m10']/(M['m00']+0.001))
#             Cy=int(M['m01']/(M['m00']+0.001))
#             
            
            
            box = np.int0(cv2.boxPoints(rect))
            
            
            xx1=abs(box[0][0]-box[1][0])
            yy1=abs(box[0][1]-box[1][1])
            A=math.sqrt(pow(xx1,2)+pow(yy1,2))
            xx2=abs(box[1][0]-box[2][0])
            yy2=abs(box[1][1]-box[2][1])
            B=math.sqrt(pow(xx2,2)+pow(yy2,2))
    #         frame2=cv2.circle(frame2,(Cx,Cy),2,(36,255,12),2)
    #         print(A/odnos,B/odnos)
    #         for x in geom_tela:
    # 
    #             x_dim_pom=x["dimenzije"]
    # 
    #             for y in x["dimenzije"]:
    # 
    #                 if real2ref(A/odnos,y,20):
    #                     
    #                     x_dim_pom.remove(y)
    # 
    #                     for i in x_dim_pom:
    #                         print(A/odnos,B/odnos)
    #                         if real2ref(B/odnos,i,10):
    # 
    #                             print(x["model"])
            stranice=pretrazivanje(A/odnos,B/odnos,7)
#             print(stranice)
            if(stranice!=None):
                cv2.drawContours(frame2, [box], 0, (36,255,12),1)
            #                             real_time_object.append({'color':'blue','model':x['model']})
                        # crtanje svih tacaka cetvororougla
                frame2=cv2.circle(frame2,(box[0][0],box[0][1]),2,(0,0,255),2)
                frame2=cv2.circle(frame2,(box[1][0],box[1][1]),2,(0,0,255),2)
                frame2=cv2.circle(frame2,(box[2][0],box[2][1]),2,(0,0,255),2)
                frame2=cv2.circle(frame2,(box[3][0],box[3][1]),2,(0,0,255),2)
                
                ax=box[0][0]
                ay=box[0][1]
                bx=box[1][0]
                by=box[1][1]
                cx=box[2][0]
                cy=box[2][1]
                Cx=abs(ax-cx)/2+min(ax,cx)
                Cy=abs(ay-cy)/2+min(ay,cy)
                if distanca(ax,ay,bx,by)<distanca(bx,by,cx,cy):
                    alpha2=-math.atan2((ay-by),(ax-bx))
                else:
                    alpha2=-math.atan2((by-cy),(bx-cx))
                
                a.append(json_create(round(Cx/odnos-pomerajX,2),round(Cy/odnos-pomerajY,2),stranice[2],round(alpha2*(180/math.pi),2),stranice[3],stranice[4],stranice[5],'c'))
                
                # crtanje ugla 
                l=40
                Px=int(Cx+l*math.cos( -alpha2))
                Py=int(Cy+l*math.sin(-alpha2))
                cv2.line(frame2,(int(Cx),int(Cy)),(Px,Py),(36,255,12),2)
#         tela=a
        
        if(len(a)==1):
            tela=a
            niz1=a
        else:
            tela=niz1
        
        json.dumps(tela)
        tela=str(tela).replace("'","\"")
#         print(len(a)!=0)
       
#         print("__________")
#         print(tela)
    #         except:
    #             pass
    #     frame1 = imutils.resize(frame1, width=480)
    #     frame2 = imutils.resize(frame2, width=1920)
    
        cv2.imshow('webcam', frame2)
        
        
    #     cv2.imshow('picam', res_top)
        
        key = cv2.waitKey(1) & 0xFF
        
        if key == 27:
            ok=False
    #             break

    cv2.destroyAllWindows()       
                
    print("[INFO] cleaning up...")

#     webcam.stop()
    picam.stop()   

    
t1=threading.Thread(target=comm)
t2=threading.Thread(target=kamere)

t2.start()
t1.start()



