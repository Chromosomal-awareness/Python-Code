import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches
from skimage.measure import label, regionprops
from skimage.color import label2rgb

def plot(grayHist):
    plt.plot(range(256), grayHist, 'r', linewidth=1.5, c='red')
    y_maxValue = np.max(grayHist)
    plt.axis([0, 255, 0, y_maxValue]) # x和y的范围
    plt.xlabel("gray Level")
    plt.ylabel("Number Of Pixels")
    plt.show()

def Show(img,title):
    plt.title(title)
    plt.imshow(img,cmap='gray')
    plt.show()

def UpperRightNode(minr,minc,maxr,maxc): #處理(maxc,minr)
    count1 = 0 #作為是否有修正的紀錄
    print('UpperRightNode')
    while True:
        count2 = 0 #作為此次是否修正完的依據
        for x in range(maxc,minc-1,-1):
            if (minr-1) >=0 and gray2[minr-1][x]<200:
                minr = minr-1
                count1 = 1
                count2 = 1
                break
        if count2 == 0:
            break
    while True:
        count2 = 0
        for y in range(minr,maxr+1):
            if (maxc+1)<gray2.shape[1] and gray2[y][maxc+1]<200:
                maxc = maxc+1
                count1 = 1
                count2 = 1
                break
        if count2 == 0:
            break
    return minr,maxc,count1    

def UpperLeftNode(minr,minc,maxr,maxc): #處理(minc,minr)
    #print('UpperLeftNode')
    count1 = 0
    while True:
        count2 = 0 #作為此次是否修正完的依據
        for x in range(minc,maxc+1):
            print(x)
            if (minr-1) >=0 and gray2[minr-1][x]<250:
                minr = minr-1
                count1 = 1
                count2 = 1
                break
        if count2 == 0:
            break
    while True:
        count2 = 0
        for y in range(minr,maxr+1):
            if (minc-1)>=0 and gray2[y][minc-1]<250:
                minc = minc-1
                count1 = 1
                count2 = 1
        if count2 == 0:
            break
    return minr,minc,count1

def LowerRightNode(minr,minc,maxr,maxc,gray2): #處理 (maxc,maxr)
    print('LowerRightNode')
    count1 = 0
    while True:
        count2 = 0
        for x in range(maxc,minc-1,-1):
            if (maxr+1)<gray2.shape[0] and gray2[maxr+1][x]<253:
                maxr = maxr+1
                count1 = 1
                count2 = 1
                break
        if count2 == 0:
            break
    while True:
        count2 = 0
        for y in range(maxr,minr-1,-1):
            if (maxc+1)<gray2.shape[1] and gray2[y][maxc+1]<253:
                maxc = maxc+1
                count1 = 1
                count2 = 1
                break
        if count2 == 0:
            break
    
    return maxr,maxc,count1
            

def LowerLeftNode(minr,minc,maxr,maxc,gray2): #處理(minc,maxr)
    print('LowerLeftNode')
    count1 = 0
    while True:
        count2 = 0
        for x in range(minc,maxc+1):
            if (maxr+1)<gray2.shape[0] and gray2[maxr+1][x]<253:
                maxr = maxr+1
                count1 = 1
                count2 = 1
                break
        if count2 == 0:
            break
    while True:
        count2 = 0
        for y in range(maxr,minr-1,-1):
            if (minc-1)>=0 and gray2[y][minc-1] <253:
                minc = minc -1
                count1 = 1
                count2 = 1
                break
        if count2==0:
            break
        
    return maxr,minc,count1


def fix(minr, minc, maxr, maxc):
    while True:
        minr,maxc,count1 = UpperRightNode(minr,minc,maxr,maxc)
        minr,minc,count2 = UpperLeftNode(minr, minc, maxr, maxc)
        #maxr,maxc,count3 = LowerRightNode(minr, minc, maxr, maxc,gray2)
        #maxr,minc,count4 = LowerLeftNode(minr, minc, maxr, maxc,gray2)
        #temp = count1 + count2 + count3 + count4
        if (count1+count2) == 0:
            break
    return minr, minc, maxr, maxc

'''    
def up(minc,minr,maxc,maxr):
    for x in range(minc,maxc+1):
        for y in range
'''
  
#file = r'Test/21.jpg'
file = r'data/1.jpg'
img = cv2.imread(file)
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
Show(gray,'0')
'''
grayHist = np.zeros(256)
h,w = gray.shape
for i in range(h):
    for j in range(w):
        grayHist[img[i][j]] += 1
plot(grayHist)
'''
for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if 0<=gray[i][j] and gray[i][j]<250:
                    gray[i][j]=0
            else:
                    gray[i][j] = 255
#Show(gray,temp+'_t0')
# 1 二質化
ret,thresh1 = cv2.threshold(gray,250,255,cv2.THRESH_BINARY_INV)
#ret2,thresh2 = cv2.threshold(gray,251,255,cv2.THRESH_BINARY)
#這裡有一個關鍵 : cv2.THRESH_BINARY_INV後續在標記時會需要他產生的效果
Show(thresh1,'1')
#Show(thresh2,'1.5')
#雜訊大約是253,254
# 2 閉運算
kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(5, 5))
close = cv2.morphologyEx(gray,cv2.MORPH_CLOSE, kernel)
Show(close,'2')

'''
#侵蝕
kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(5, 5))
erosion = cv2.erode(thresh1,kernel,iterations = 1)
erosion2 = cv2.erode(thresh2,kernel,iterations = 1)
Show(erosion,'2')
Show(erosion2,'2.5')
#膨脹
kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(5, 5))
dilation = cv2.dilate(erosion,kernel,iterations = 1)
dilation2 = cv2.dilate(erosion2,(3,3),iterations = 1)
Show(dilation,'3')
Show(dilation2,'3.5')
'''
#自動適應二質化
adaptive_thresh = cv2.adaptiveThreshold(close, 255, 
                                        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                        cv2.THRESH_BINARY_INV, 
                                        3, 1)
Show(adaptive_thresh,'3')

# 4 標記圖型
label_img = label(adaptive_thresh, connectivity = 1) #連通區域
image_label_overlay = label2rgb(label_img,image=img) #標記
print('連通區域 : ',label_img.max()+1)
prop = regionprops(label_img)
fig, ax = plt.subplots(figsize=(16, 12))
for region in prop:
    if region.area >= 180: #300
        minr, minc, maxr, maxc = region.bbox
        #print(region.bbox)
        #tuple1 = ( minc, minr, maxc, maxr)
        #print('處理前 : ',tuple1)
        minr, minc, maxr, maxc = fix(minr, minc, maxr, maxc)
        #tuple1 = ( minc, minr, maxc, maxr)
        #print('處理後 : ',tuple1)
        #print('=========================================')
        rect = patches.Rectangle((minc,minr), maxc - minc, maxr - minr,
                                 fill=False,edgecolor='red', linewidth=0.5)
        #patches用於生成圖形
        ax.add_patch(rect)
        #用add_patch方法加入圖片
    plt.title('new_version')
    ax.imshow(img,cmap='gray')
