# Import required packages 
import cv2 
import pytesseract 
import matplotlib.pyplot as plt
import numpy as np

# Mention the installed location of Tesseract-OCR in your system 
pytesseract.pytesseract.tesseract_cmd = r"E:\Program Files\Tesseract-OCR\tesseract"

# Read image from which text needs to be extracted 
image = cv2.imread("mypan.jpg") 
Plot(image)

#ploting the image
def Plot(image):
    plt.imshow(image,'gray')
    plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
    plt.show()
    
#invert image
def Invert(image):
    return (cv2.bitwise_not(image))

#convert to gray
imageGray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
Plot(imageGray)

ret2, thresh2 = cv2.threshold(imageGray, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV) 
Plot(thresh2)

#we will use getStructuringElement to capture words.
#we will specify the maximum filter/kernel size
#and shape here we are choosing rectangle filter with size of 18 by 18 units.
rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (18, 18)) 

# Appplying dilation on the threshold image 
# it means it will expand the white region by number if times in the iteration.
dilation = cv2.dilate(thresh2, rect_kernel, iterations = 1) 
Plot(dilation)

cleaned = cv2.morphologyEx(dilation, cv2.MORPH_OPEN, rect_kernel)
Plot(cleaned)

# Finding contours 
contours, hierarchy = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, 
												cv2.CHAIN_APPROX_NONE) 
#draw countours in an image
#cv2.drawContours(image, contours, -1, (0,255,0), 3)
#Plot(image)

# Creating a copy of image 
im2 = image.copy() 

# Looping through the identified contours 
# Then rectangular part is cropped and passed on 
# to pytesseract for extracting text from it 
# Extracted text is then written into the text file 
infromation = {}

for cnt in contours:
    x, y, w, h = cv2.boundingRect(cnt) 
	
	# Drawing a rectangle on copied image 
    rect = cv2.rectangle(im2, (x, y), (x + w, y + h), (0, 255, 0), 15) 
    Plot(rect)

    # Cropping the text block for giving input to OCR 
    cropped = im2[y:y + h, x:x + w]
    
    #convert to gray
    croppedImageGray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
    #Plot(croppedImageGray)
	
	# Apply OCR on the cropped image
    config = ('-l eng --oem 3 --psm 11')
    text = pytesseract.image_to_string(croppedImageGray,config = config)
    if(len(text) >= 3):
        print(text)
        Plot(croppedImageGray)
        
        
