import numpy as np
from cv2 import data
from PyQt5.QtGui import QPixmap
from  PyQt5.QtWidgets import  QMainWindow, QApplication,QLabel,QPushButton
from PyQt5 import uic
import sys
import cv2
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *

class UI(QMainWindow):
    def __init__(self):
        super(UI,self).__init__()
        uic.loadUi('cartoon.ui',self)
        self.label = self.findChild(QLabel,"label")
        self.image_label = self.findChild(QLabel,"image1")
        self.cartoon_label = self.findChild(QLabel, "cartoon")
        self.save_btn = self.findChild(QPushButton, "save_btn")
        self.cartoon_image_btn = self.findChild(QPushButton, "image")
        self.edge_btn = self.findChild(QPushButton, "edge_btn")
        self.final =""
        self.mask = ""
        self.show()

        self.cartoon_image_btn.clicked.connect(self.cartoonify)
        self.save_btn.clicked.connect(self.save)
        self.edge_btn.clicked.connect( self.edge_detect)


    def cartoonify(self):
        fname = QFileDialog.getOpenFileName(self, 'Open file',
                                            '', "Image files (*.jpg *.gif)")
        self.pixmap_size = QSize(self.image_label.width(),self.image_label.height())
        self.pixmap = QPixmap(fname[0])
        self.pixmap=self.pixmap.scaled(self.pixmap_size)
        self.image_label.setPixmap(self.pixmap)
        imagepath = fname[0]

        img = cv2.imread(imagepath)
        # convert image to RGB format
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        #convert the image to grayscale
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

        # Median Blurring
        img_blur = cv2.medianBlur(gray,5)

        # Creating Edge Mask
        edges = cv2.adaptiveThreshold(img_blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,3 ,3)

        # Removing Noise
        img_n = cv2.bilateralFilter(img_blur, 15, 75, 75)

        cartoonImage = cv2.bitwise_and( img_n,  img_n, mask=edges)
        cartoonImage = cv2.cvtColor(cartoonImage, cv2.COLOR_BGR2RGB)
        kernel = np.ones((1, 1), np.uint8)

        # Eroding and Dilating
        img_erode = cv2.erode(img_n, kernel, iterations=10)
        img_dilate = cv2.dilate(img_erode, kernel, iterations=3)

        #Stylization on the image
        final=cv2.stylization(img, sigma_s= 150 , sigma_r=0.25)


        #Clustering - (K-means)
        imgf = np.float32(img).reshape((-1, 3))
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.001 )
        compactness, label, center = cv2.kmeans(imgf, 5, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        center = np.uint8(center)
        final_img = center[label.flatten()]
        final_img = final_img.reshape(img.shape)
        self.final = cv2.bitwise_and(final_img,final_img, mask=edges)
        self.final = cv2.cvtColor(self.final, cv2.COLOR_BGR2RGB)
        cv2.imwrite( 'cartoon.jpg', self.final)
        self.pixmap_detect_size = QSize(self.cartoon_label.width(), self.cartoon_label.height())
        self.pixmap_detect = QPixmap('cartoon.jpg')
        self.pixmap_detect = self.pixmap_detect.scaled(self.pixmap_detect_size)
        self.cartoon_label.setPixmap(self.pixmap_detect)
        # cv2.imshow('final', final)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

    def edge_detect(self):
        fname = QFileDialog.getOpenFileName(self, 'Open file',
                                            '', "Image files (*.jpg *.gif)")
        self.pixmap_size = QSize(self.image_label.width(),self.image_label.height())
        self.pixmap = QPixmap(fname[0])
        self.pixmap=self.pixmap.scaled(self.pixmap_size)
        self.image_label.setPixmap(self.pixmap)
        imagepath = fname[0]
        img = cv2.imread(imagepath)

        #convert the image to grayscale
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        cv2.imshow('ad', gray)
        # Median Blurring
        img_blur = cv2.GaussianBlur(gray, (5, 5), 0)
        canny = cv2.Canny(img_blur, 10, 70)
        ret, self.final= cv2.threshold(canny, 70, 255, cv2.THRESH_BINARY)


        cv2.imwrite( 'edge.jpg', self.final)
        self.pixmap_detect_size = QSize(self.cartoon_label.width(), self.cartoon_label.height())
        self.pixmap_detect = QPixmap('edge.jpg')
        self.pixmap_detect = self.pixmap_detect.scaled(self.pixmap_detect_size)
        self.cartoon_label.setPixmap(self.pixmap_detect)
        # cv2.imshow('final', final)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
    def save(self):

        # selecting file path
        filePath, _ = QFileDialog.getSaveFileName(self, "Save Image", "",
                                                  "PNG(*.png);;JPEG(*.jpg *.jpeg);;All Files(*.*) ")
        # if file path is blank return back
        if filePath == "":
            return

        # saving canvas at desired path
        cv2.imwrite( filePath,self.final)


if __name__ == '__main__':
    app=QApplication(sys.argv)
    UIWindow = UI()
    app.exec_()

