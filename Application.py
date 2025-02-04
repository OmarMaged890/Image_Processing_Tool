import sys
import os
import cv2
import numpy as np
from PyQt5.QtWidgets import QApplication, QFileDialog, QDialog, QPushButton, QGraphicsView, QGraphicsScene, QLabel, QSlider
from PyQt5 import uic
from PyQt5.QtGui import QPixmap, QImage

class ImageProcessingApp(QDialog):
    def __init__(self):
        super(ImageProcessingApp, self).__init__()

        # Load UI
        ui_path = os.path.join(os.path.dirname(__file__), "app.ui")
        uic.loadUi(ui_path, self)

        # Load Image Button
        self.load_button = self.findChild(QPushButton, 'pushButton')
        self.load_button.clicked.connect(self.load_image)

        # Edge Detection Buttons
        self.prewitt_button = self.findChild(QPushButton, 'pushButton_2')
        self.prewitt_button.clicked.connect(self.apply_prewitt)
        
        self.roberts_button = self.findChild(QPushButton, 'pushButton_3')
        self.roberts_button.clicked.connect(self.apply_roberts)

        self.canny_button = self.findChild(QPushButton, 'pushButton_4')
        self.canny_button.clicked.connect(self.apply_canny)

        self.log_button = self.findChild(QPushButton, 'pushButton_5')
        self.log_button.clicked.connect(self.apply_log)

        # Segmentation Buttons
        self.hist_seg_button = self.findChild(QPushButton, 'pushButton_8')
        self.hist_seg_button.clicked.connect(self.histogram_segmentation)

        self.manual_seg_button = self.findChild(QPushButton, 'pushButton_7')
        self.manual_seg_button.clicked.connect(self.manual_segmentation)

        self.adaptive_hist_button = self.findChild(QPushButton, 'pushButton_9')
        self.adaptive_hist_button.clicked.connect(self.adaptive_histogram_segmentation)

        self.otsu_button = self.findChild(QPushButton, 'pushButton_10')
        self.otsu_button.clicked.connect(self.otsu_segmentation)

        # Gaussian Blur Slider
        self.gaussian_slider = self.findChild(QSlider, 'horizontalSlider')
        self.gaussian_slider.setMinimum(1)
        self.gaussian_slider.setMaximum(15)  # Kernel size range
        self.gaussian_slider.setValue(5)
        self.gaussian_slider.setSingleStep(2)  # Ensure odd values
        self.gaussian_slider.valueChanged.connect(self.apply_gaussian_filter)

        # Median Blur Slider
        self.median_slider = self.findChild(QSlider, 'horizontalSlider_2')
        self.median_slider.setMinimum(1)
        self.median_slider.setMaximum(15)
        self.median_slider.setValue(5)
        self.median_slider.setSingleStep(2)
        self.median_slider.valueChanged.connect(self.apply_median_filter)

        # Smoothing Slider
        self.smoothing_slider = self.findChild(QSlider, 'horizontalSlider_3')
        self.smoothing_slider.setMinimum(1)
        self.smoothing_slider.setMaximum(15)
        self.smoothing_slider.setValue(5)
        self.smoothing_slider.setSingleStep(2)
        self.smoothing_slider.valueChanged.connect(self.apply_smoothing)

        # Translation Slider
        self.translation_slider = self.findChild(QSlider, 'horizontalSlider_4')
        self.translation_slider.setMinimum(-100)
        self.translation_slider.setMaximum(100)
        self.translation_slider.setValue(0)
        self.translation_slider.valueChanged.connect(self.apply_translation)

        # Rotation Slider
        self.rotation_slider = self.findChild(QSlider, 'horizontalSlider_5')
        self.rotation_slider.setMinimum(-180)
        self.rotation_slider.setMaximum(180)
        self.rotation_slider.setValue(0)
        self.rotation_slider.valueChanged.connect(self.apply_rotation)
        
        # Reset Button
        self.reset_button = self.findChild(QPushButton, 'pushButton_11')
        self.reset_button.clicked.connect(self.reset_image)

        # Image Viewer
        self.image_viewer = self.findChild(QGraphicsView, 'graphicsView_3')
        self.scene = QGraphicsScene()
        self.image_viewer.setScene(self.scene)

        self.image = None  # Store the loaded image
        self.processed_image = None  # Store processed images

    def load_image(self):
        fname, _ = QFileDialog.getOpenFileName(self, "Open Image File", "", "Images (*.png *.jpg *.jpeg *.bmp)")
        if fname:
            self.image = cv2.imread(fname)
            self.processed_image = self.image.copy()
            self.display_image()
            
    def apply_roberts(self):
        if self.image is None:
            return
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        kernel_x = np.array([[1, 0], [0, -1]])
        kernel_y = np.array([[0, 1], [-1, 0]])
        roberts_x = cv2.filter2D(gray, -1, kernel_x)
        roberts_y = cv2.filter2D(gray, -1, kernel_y)
        self.processed_image = cv2.addWeighted(roberts_x, 0.5, roberts_y, 0.5, 0)
        self.display_image()

    def apply_prewitt(self):
        if self.image is None:
            return
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        prewitt_x = cv2.filter2D(gray, -1, np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]]))
        prewitt_y = cv2.filter2D(gray, -1, np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]]))
        self.processed_image = cv2.addWeighted(prewitt_x, 0.5, prewitt_y, 0.5, 0)
        self.display_image()

    def apply_canny(self):
        if self.image is None:
            return
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        self.processed_image = cv2.Canny(gray, 100, 200)  # Canny Edge Detection
        self.display_image()

    def apply_log(self):
        if self.image is None:
            return
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        laplacian = cv2.Laplacian(blurred, cv2.CV_64F)  # LoG output in float64

        # Convert to 8-bit unsigned integer (CV_8U)
        self.processed_image = cv2.convertScaleAbs(laplacian)  
        self.display_image()


    def display_image(self, img=None):
        if img is None:
            img = self.processed_image
        
        if img is None:
            return

        image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, ch = image.shape
        bytes_per_line = ch * w
        q_image = QImage(image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        
        self.scene.clear()
        self.scene.addPixmap(QPixmap.fromImage(q_image))
        self.image_viewer.setScene(self.scene)

    def reset_image(self):
        if self.image is not None:
            self.processed_image = self.image.copy()
            self.display_image()

    def histogram_segmentation(self):
        if self.image is None:
            return
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        threshold = np.argmax(hist)
        _, self.processed_image = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
        self.display_image()

    def manual_segmentation(self):
        if self.image is None:
            return
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        _, self.processed_image = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)
        self.display_image()

    def adaptive_histogram_segmentation(self):
        if self.image is None:
            return
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        self.processed_image = cv2.equalizeHist(gray)
        self.display_image()

    def otsu_segmentation(self):
        if self.image is None:
            return
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        _, self.processed_image = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        self.display_image()
        
    def apply_gaussian_filter(self):
        if self.image is None:
            return
        ksize = self.gaussian_slider.value()
        if ksize % 2 == 0:
            ksize += 1  # Ensure odd kernel size
        self.processed_image = cv2.GaussianBlur(self.image, (ksize, ksize), 0)
        self.display_image()

    def apply_median_filter(self):
        if self.image is None:
            return
        ksize = self.median_slider.value()
        if ksize % 2 == 0:
            ksize += 1
        self.processed_image = cv2.medianBlur(self.image, ksize)
        self.display_image()

    def apply_smoothing(self):
        if self.image is None:
            return
        ksize = self.smoothing_slider.value()
        if ksize % 2 == 0:
            ksize += 1
        self.processed_image = cv2.blur(self.image, (ksize, ksize))
        self.display_image()

    def apply_translation(self):
        if self.image is None:
            return
        h, w = self.image.shape[:2]
        shift = self.translation_slider.value()
        translation_matrix = np.float32([[1, 0, shift], [0, 1, shift]])
        self.processed_image = cv2.warpAffine(self.image, translation_matrix, (w, h))
        self.display_image()
        
    def apply_rotation(self):
        if self.image is None:
            return
        angle = self.rotation_slider.value()
        h, w = self.image.shape[:2]
        rotation_matrix = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1)
        self.processed_image = cv2.warpAffine(self.image, rotation_matrix, (w, h))
        self.display_image()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ImageProcessingApp()
    window.show()
    sys.exit(app.exec_()) 