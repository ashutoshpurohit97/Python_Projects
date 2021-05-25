import cv2
import numpy as np
import matplotlib.pyplot as plt
import colorsys
import math


def main():

    path = "D:\Set study\processed\\"
    img1=path+"3.jpg"
    img2=path+"3.jpg"
    original = cv2.imread(img1)
    image_to_compare = cv2.imread(img2)

    
    print("RGB color space")
    img1 = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
    img2 = cv2.cvtColor(image_to_compare, cv2.COLOR_BGR2RGB)
    plt.imshow(img1)
    plt.title('original')
    plt.xticks([])
    plt.yticks([])
    plt.show()
    
    plt.imshow(img2)
    plt.title('image_to_compare')
    plt.xticks([])
    plt.yticks([])
    plt.show()
    
    R1, G1, B1 = cv2.split(img1)
    R2, G2, B2 = cv2.split(img2)
    #print(R1)
    mean_R1=np.mean(R1)
    #print(mean_R1)
    #print(G1)
    mean_G1=np.mean(G1)
    #print(mean_G1)
    #print(B1)
    mean_B1=np.mean(B1)
    #print(mean_B1)
       
    #print(R2)
    mean_R2=np.mean(R2)
    #print(mean_R2)
    #print(G2)
    mean_G2=np.mean(G2)
    #print(mean_G2)
    #print(B2)
    mean_B2=np.mean(B2)
    #print(mean_B2)
    
    #print("Color Distance RGB values")
    Rx=(mean_R1-mean_R2)**2
    #print("R=",Rx)
    Gx=(mean_G1-mean_G2)**2
    #print("G=",Gx)
    Bx=(mean_B1-mean_B2)**2
    #print("B=",Bx)
    dx=np.sqrt(Rx+Gx+Bx)
    d=np.sqrt(mean_R1**2+mean_G1**2+mean_B1**2)
    
    if(d==0):
        accuracy= (abs(d-dx)/dx*100)
        print("Overall Accuracy is",accuracy,"%")
        
    else:
        accuracy= (abs(d-dx)/d*100)
        print("Overall Accuracy is",accuracy,"%")
        
    if((10>=mean_R1>=0) and (10>=mean_G1>=0) and (10>=mean_B1>=0)):
        print("The color of 1st image is black")
    



if __name__ == "__main__":
    main()