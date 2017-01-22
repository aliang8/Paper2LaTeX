import cv2
#from textRecognition import *
#from textRecognitionCNN import *
from textExtracter import *

file_name ='../images/e.jpg'

def findTextRegion(file_name):
    img  = cv2.imread(file_name)


    img_final = cv2.imread(file_name)
    img2gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)


    # Use Canny Edge Detection
    img2gray = cv2.medianBlur(img2gray, 5)
    new_img = cv2.Canny(img2gray,10,50)
    #cv2.imshow('Canny', new_img)
    #cv2.waitKey(0)
    # Preform morphological transform
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,(3 , 3)) 

    # TEXT REGIONS
    # dilate is to increase blob size, erode is to increase blob size
    morphed_img = cv2.dilate(new_img,kernel,iterations = 3) # dilate , more the iteration more the dilation
    morphed_img = cv2.erode(morphed_img, kernel, iterations = 3)

    cv2.imshow('morphed',morphed_img)

    _,contours, hierarchy = cv2.findContours(morphed_img,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE) # get contours
    index = 0
 
    for contour in contours:
        # get rectangle bounding contour
        [x,y,w,h] = cv2.boundingRect(contour)

	
        # Regulate size (prevent too small or too large detection)
        
	
        # draw rectangle around contour on original image
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,255),2)


        cropped = img_final[y :y +  h , x : x + w]
	
	
	
	#cv2.imshow('cropped', cropped)
	#cv2.waitKey(0)

	recognizeText(cropped)

        '''
        #you can crop image and send to OCR  , false detected will return no text :)
        cropped = img_final[y :y +  h , x : x + w]

        s = file_name + '/crop_' + str(index) + '.jpg' 
        cv2.imwrite(s , cropped)
        index = index + 1

        '''
    # write original image with added contours to disk  
    cv2.imshow('captcha_result' , img)
    cv2.waitKey()


findTextRegion(file_name)
