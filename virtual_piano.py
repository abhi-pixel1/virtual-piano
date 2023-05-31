'''
a playable virtual piano
'''


import mediapipe as mp
import cv2
import numpy as np
# import pyautogui
import imutils
import pyautogui
# from playsound import playsound


from selenium import webdriver
from webdriver_manager.chrome import ChromeDriverManager
import time
import sys


# chrome_path = r"C:/Users/bhara/py/ct/ch/chromedriver112.exe"
chrome_path = r"C:\\Users\\bhara\\py\\ct2\\chromedriver.exe"

driver = webdriver.Chrome(ChromeDriverManager().install())

url=r'https://www.apronus.com/music/flashpiano.htm'
driver.get(url)
time.sleep(3)




x_list=[]
x_dict={}



mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands


def Press(key):
    pyautogui.press(key)





def drawkeys(img,keylist):
    for k in keylist:
        x,y = k.pos
        #y = k.pos
        w,h = k.size
        #h = k.size
        colour = k.color
        cv2.rectangle(img,k.pos,(x+w,y+h),colour,1)
        cv2.putText(img,k.text,(x+10,y+h-10),cv2.FONT_HERSHEY_COMPLEX,0.5,(214,0,220),2)
    return img


cap = cv2.VideoCapture(0)
cap.set(3,1920)
cap.set(4,1080)



class Button():
    def __init__(self,pos,text,size,color):
        self.pos = pos
        self.text = text
        self.size = size
        self.color = color

#Setting piano keys
keys = [["C","D","E","F","G","A","B","C1","D1","E1","F1","G1","A1","B1"],["C#","D#","F#","G#","A#","C1#","D1#","F1#","G1#","A1#"]]

#Initializing keys and setting parameters for keys as well as coordinates
#Parameters are position, keynote, size and color
keylist = []
for i in range(len(keys)):
    for j, key in enumerate(keys[i]):
        if i == 0:
            keylist.append(Button([38*j+15, 80], key,[35, 100], (255, 255, 255)))
        else:
            keylist.append(Button([(40+j)*j+25, 80], key, [35, 50], (0, 0, 0)))


#Setting Hand Detector
# det = HandDetector(detectionCon=1)

for i in keylist:
  print(i.pos[0],i.text,i.size)
  x_dict[i.text]=[i.pos[0],(i.pos[0]+i.size[0]),i.color]

print(x_dict)





with mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands: 
  while True:
    ret, frame = cap.read()
    # frame = cv2.flip(frame,1)
    # frame = imutils.resize(frame,height=1000, width=1500)

    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
    # Flip on horizontal
    image = cv2.flip(image, 1)
    image = drawkeys(image,keylist)
    # Set flag
    image.flags.writeable = False
        
    # Detections
    results = hands.process(image)
        
    # Set flag to true
    image.flags.writeable = True
        
    # RGB 2 BGR
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)


    # image/frame, start_point, end_point, color, thickness
    # cv2.rectangle(image, (0,0), (2,150), (255,0,0),1)
    # cv2.putText(image,'RIDE',(70,80),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),3,cv2.LINE_AA)
    # cv2.rectangle(image, (210,0), (430,150), (0,0,255),1)
    # cv2.putText(image,'RIDE BELL',(245,80),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),3,cv2.LINE_AA)
    # cv2.rectangle(image, (440,0), (650,150), (255,0,0),1)
    # cv2.putText(image,'HITHAT close',(445,80),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),3,cv2.LINE_AA)
    # cv2.rectangle(image, (660,0), (900,150), (0,0,255),1)
    # cv2.putText(image,'CRASH',(730,80),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),3,cv2.LINE_AA)



    if results.multi_hand_landmarks:
        for num, hand in enumerate(results.multi_hand_landmarks):
            mp_drawing.draw_landmarks(image, hand, mp_hands.HAND_CONNECTIONS, 
                                    mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
                                    mp_drawing.DrawingSpec(color=(250, 44, 250), thickness=2, circle_radius=2),
                                      )
            # print(num)
            # print(hand)

            coords = tuple(np.multiply(
            np.array((hand.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x, hand.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y)),
            [1080,700]).astype(int))
            # print(coords)



            cv2.putText(image,'hello',coords,cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

            x = coords[0]
            y = coords[1]
            if x > 15 and y > 80 and x < 509 and y < (80+100): 
              # playsound(r'C:\Users\bhara\Downloads\D.wav')
              cv2.putText(image,'namo',(200,49),cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

              if x>x_dict['C'][0] and x<x_dict['C'][1]:
                keyd = driver.find_element_by_xpath('//*[@id="klawisz48"]')
                keyd.click()

              elif x>x_dict['D'][0] and x<x_dict['D'][1]:
                keyd = driver.find_element_by_xpath('//*[@id="klawisz50"]')
                keyd.click()

              elif x>x_dict['E'][0] and x<x_dict['E'][1]:
                keyd = driver.find_element_by_xpath('//*[@id="klawisz52"]')
                keyd.click()

              elif x>x_dict['F'][0] and x<x_dict['F'][1]:
                keyd = driver.find_element_by_xpath('//*[@id="klawisz53"]')
                keyd.click()

              elif x>x_dict['G'][0] and x<x_dict['G'][1]:
                keyd = driver.find_element_by_xpath('//*[@id="klawisz55"]')
                keyd.click()

              elif x>x_dict['A'][0] and x<x_dict['A'][1]:
                keyd = driver.find_element_by_xpath('//*[@id="klawisz57"]')
                keyd.click()

              elif x>x_dict['B'][0] and x<x_dict['B'][1]:
                keyd = driver.find_element_by_xpath('//*[@id="klawisz59"]')
                keyd.click()

              elif x>x_dict['C1'][0] and x<x_dict['C1'][1]:
                keyd = driver.find_element_by_xpath('//*[@id="klawisz60"]')
                keyd.click()

              elif x>x_dict['D1'][0] and x<x_dict['D1'][1]:
                keyd = driver.find_element_by_xpath('//*[@id="klawisz62"]')
                keyd.click()

              elif x>x_dict['E1'][0] and x<x_dict['E1'][1]:
                keyd = driver.find_element_by_xpath('//*[@id="klawisz64"]')
                keyd.click()     

              elif x>x_dict['F1'][0] and x<x_dict['F1'][1]:
                keyd = driver.find_element_by_xpath('//*[@id="klawisz65"]')
                keyd.click()

              elif x>x_dict['G1'][0] and x<x_dict['G1'][1]:
                keyd = driver.find_element_by_xpath('//*[@id="klawisz67"]')
                keyd.click()

              elif x>x_dict['A1'][0] and x<x_dict['A1'][1]:
                keyd = driver.find_element_by_xpath('//*[@id="klawisz69"]')
                keyd.click()
                break


    frame = cv2.flip(frame,1)
    cv2.imshow('Hand Tracking', image)
    if cv2.waitKey(10) & 0xFF == ord('q'):
      break

cap.release()
cv2.destroyAllWindows()