import csv
import os
import sys
import cv2
import dlib
import time
import numpy as np
import pandas as pd

from datetime import datetime
from keras.models import load_model

from PyQt5 import QtWidgets, QtGui, QtCore
from PyQt5.QtCore import pyqtSlot, QRunnable, pyqtSignal, QObject, QThreadPool
from PyQt5.QtGui import QImage
from numpy import expand_dims
from sklearn.ensemble import RandomForestClassifier

from GUI import Ui_MainWindow
from Warning import Ui_Dialog


# The Main class
class Main:

    # The Constructor
    def __init__(self):

        # Global Variables
        self.embeddings = []
        self.model = RandomForestClassifier()
        self.images = []
        self.images_folder = "Captured images"
        self.camera_id = 0

        # thread pool
        self.threadpool = QThreadPool()

        # labels for model training
        self.classes = []

        # loading the camera
        self.video = cv2.VideoCapture(self.camera_id)

        # loading the face detector
        self.detector = dlib.get_frontal_face_detector()

        # count of the frames
        self.count = 0

        # default image size
        self.image_size = 160

        # camera stream control flag
        self.camera_stream = False

        # Flag to capture face or not
        self.capture_flag = False

        # The pretrained model to extract embeddings
        self.emb_model = load_model('facenet_keras.h5', compile=False)

        # Output dataframe time
        if os.path.exists("report.csv"):
            self.report = pd.read_csv('report.csv')
        else:
            self.report = pd.DataFrame(columns=['ID', 'Name', 'Date', 'Time'])

        # Sample Amount to take
        self.samples_amount = 100

        # Creating the main window object
        self.main_window = QtWidgets.QMainWindow()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self.main_window)

        # For warning dialogue
        self.warning_window = QtWidgets.QDialog()
        self.warning_obj = Ui_Dialog()
        self.warning_obj.setupUi(self.warning_window)
        self.warning_obj.pushButton.clicked.connect(lambda: self.warning_window.close())

        # All button connections
        self.ui.pushButton.clicked.connect(lambda: self.ui.stackedWidget.setCurrentWidget(self.ui.page_3))
        self.ui.pushButton_9.clicked.connect(lambda: self.ui.stackedWidget.setCurrentWidget(self.ui.page_2))
        self.ui.pushButton_3.clicked.connect(lambda: self.ui.stackedWidget.setCurrentWidget(self.ui.page_4))
        self.ui.pushButton_10.clicked.connect(lambda: self.ui.stackedWidget.setCurrentWidget(self.ui.page))
        self.ui.pushButton_12.clicked.connect(lambda: self.ui.stackedWidget.setCurrentWidget(self.ui.page))
        self.ui.pushButton_13.clicked.connect(lambda: self.ui.stackedWidget.setCurrentWidget(self.ui.page))
        self.ui.pushButton_11.clicked.connect(lambda: self.ui.stackedWidget.setCurrentWidget(self.ui.page))

        # Connecting buttons to functions
        self.ui.pushButton_5.clicked.connect(self.capture_flag_fun)
        self.ui.pushButton_6.clicked.connect(self.capture_faces)
        self.ui.pushButton_8.clicked.connect(self.start_tracking_on_camera)
        self.ui.pushButton_4.clicked.connect(self.view_Report)
        self.ui.pushButton_7.clicked.connect(self.upload_image_from_gallery)
        self.ui.pushButton_2.clicked.connect(self.recognize_face_on_image)

        # Start with initial page
        self.ui.stackedWidget.setCurrentWidget(self.ui.page)

    def recognize_face_on_image(self):

        # Load the names
        names_dict = pd.read_csv("names.csv").values[0]

        # open the dialogue box to select the file
        options = QtWidgets.QFileDialog.Options()

        # open the Dialogue box to get the images paths
        images = QtWidgets.QFileDialog.getOpenFileName(caption="Select the image", directory="",
                                                       filter="Image Files (*.jpg);;Image Files (*.png)",
                                                       options=options)

        # Check if the images are taken or not
        if len(images) == 0:
            self.warning_obj.label_3.setText("No images provided")
            self.warning_window.show()
            return

        # Start training
        f = self.start_training()
        if f:
            return
        image = cv2.imread(images[0])
        # convert the frame into grayscale and get the face Coordinates
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.detector(gray)

        # Check face or not found
        if len(faces) != 0:

            # Now get all the face in the frame
            for face in faces:
                x1 = face.left()
                y1 = face.top()
                x2 = face.right()
                y2 = face.bottom()

                # Now get the reign of interest of the face and get the prediction over that face
                roi = image[y1:y2, x1:x2]

                # Images resizing and preprocessing
                try:
                    roi = cv2.resize(roi, (self.image_size, self.image_size))
                except:
                    continue

                emb = self.get_embedding(roi)
                pred = self.model.predict([emb])[0]

                # Draw a green box over the face
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.rectangle(image, (x2 - (x2 - x1), y2), (x2, y2 + 50), (0, 255, 0), -1)

                # Display the id text
                cv2.putText(image, str(names_dict[pred]), (x1, y2 + 30), cv2.FONT_HERSHEY_SIMPLEX, 1,
                            (255, 255, 255), 2)

                result = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                height, width, channel = result.shape
                step = channel * width
                qImg = QImage(result.data, width, height, step, QImage.Format_RGB888)
                self.ui.label_7.setPixmap(QtGui.QPixmap(qImg))

    def upload_image_from_gallery(self):

        if self.ui.lineEdit_2.text() == "":
            self.warning_obj.label_3.setText("Please enter person name")
            self.warning_window.show()
            return

        if self.ui.pushButton_7.text() == "Upload Images":

            self.ui.pushButton_7.setText("Uploading Images")

            # open the dialogue box to select the file
            options = QtWidgets.QFileDialog.Options()

            # open the Dialogue box to get the images paths
            images = QtWidgets.QFileDialog.getOpenFileNames(caption="Select the images of the subject", directory="",
                                                            filter="Image Files (*.jpg);;Image Files (*.png)",
                                                            options=options)

            # Check if the images are taken or not
            if len(images[0]) == 0:
                self.warning_obj.label_3.setText("No images provided")
                self.warning_window.show()
                return

            # Reset the images counter
            self.count = 0

            # Traverse over all the images path
            for image_path in images[0]:

                # Read the images
                image = cv2.imread(image_path)

                # convert the frame into grayscale and get the face Coordinates
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

                # Getting the face coordinates from the image
                faces = self.detector(gray)

                # Check face or not found
                if len(faces) != 0:

                    # Now get all the face in the frame
                    for face in faces:
                        x1 = face.left()
                        y1 = face.top()
                        x2 = face.right()
                        y2 = face.bottom()

                        # Now get the reign of interest of the face and get the prediction over that face
                        roi = image[y1:y2, x1:x2]

                        # Images resizing
                        try:
                            cv2.resize(roi, (self.image_size, self.image_size))
                        except:
                            continue

                        # Getting the images
                        self.images.append(roi)

                        # Increment the count
                        self.count += 1

                        if self.count == self.ui.comboBox_6.currentText():
                            break

            # Saving face images
            if self.images != []:
                # Extract the person name
                person_name = self.ui.lineEdit_2.text()

                # Create the id named folder for the image
                if not os.path.exists(os.path.join(self.images_folder, person_name)):
                    os.mkdir(os.path.join(self.images_folder, person_name))

                # Save the face images
                for c, img in enumerate(self.images):
                    cv2.imwrite(os.path.join(self.images_folder, person_name, person_name + '.' + str(c) + '.jpg'), img)

            # Empty the image list
            self.images = []

            self.ui.pushButton_7.setText("Upload Images")

    def capture_flag_fun(self):
        if self.ui.pushButton_5.text() == "Start Capture Face":
            if self.ui.pushButton_6.text() == "Start Camera":
                self.warning_obj.label_3.setText("Please start the camera first")
                self.warning_window.show()
                return
            self.capture_flag = True
            self.ui.pushButton_5.setText("Stop Capture Face")
        else:
            self.capture_flag = False
            self.ui.pushButton_5.setText("Start Capture Face")

    def capture_faces(self):

        if self.ui.lineEdit.text() == "":
            self.warning_obj.label_3.setText("Please enter person name")
            self.warning_window.show()
            return

        if self.ui.pushButton_6.text() == "Start Camera":

            self.video = cv2.VideoCapture(self.camera_id)
            # Set the button text to stop camera
            self.ui.pushButton_6.setText("Stop Camera")

            # Set the camera stream flag
            self.camera_stream = True

            # Updating the current epochs value
            self.samples_amount = int(self.ui.comboBox_5.currentText())

            # Resetting count
            self.count = 0

            # Loop for the camera frames
            while True:
                # Get the frame from camera
                ret, frame = self.video.read()

                # Resizing the input frame
                frame = cv2.resize(frame, (500, 400))

                # if frame is read
                if ret:
                    # convert the frame into grayscale and get the face Coordinates
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                    # Getting the face coordinates from the image
                    faces = self.detector(gray)

                    # Check face or not found
                    if len(faces) != 0:

                        # Now get all the face in the frame
                        for face in faces:
                            x1 = face.left()
                            y1 = face.top()
                            x2 = face.right()
                            y2 = face.bottom()

                            # Now get the reign of interest of the face and get the prediction over that face
                            roi = frame[y1:y2, x1:x2]

                            # Images resizing
                            try:
                                cv2.resize(roi, (self.image_size, self.image_size))
                            except:
                                continue

                            if self.capture_flag:
                                # Getting the images
                                self.images.append(roi)

                                # Increment the count
                                self.count += 1

                            # Draw a green box over the face
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                    # Show the count of faces Captured
                    cv2.putText(frame, "Face Captured:" + str(self.count), (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                (0, 255, 0), 2)

                    # Show the image
                    result = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    height, width, channel = result.shape
                    step = channel * width
                    qImg = QImage(result.data, width, height, step, QImage.Format_RGB888)
                    self.ui.label_3.setPixmap(QtGui.QPixmap(qImg))

                    cv2.waitKey(1)

                    # Break the loop for the 100th frame
                    if self.count == self.samples_amount:
                        break

                    # break if the camera stream is null
                    if not self.camera_stream:
                        break

        elif self.ui.pushButton_6.text() == "Stop Camera":
            self.camera_stream = False
            self.ui.pushButton_6.setText("Start Camera")
            return

        # Release the camera and destroy the windows
        self.video.release()
        self.reset_windows()

        if self.images != []:
            # Extract the person name
            person_name = self.ui.lineEdit.text()

            # Create the id named folder for the image
            if not os.path.exists(os.path.join(self.images_folder, person_name)):
                os.mkdir(os.path.join(self.images_folder, person_name))

            # Save the face images
            for c, img in enumerate(self.images):
                cv2.imwrite(os.path.join(self.images_folder, person_name, person_name + '.' + str(c) + '.jpg'), img)

            # Empty the image list
            self.images = []

        # Now the image capturing is off
        self.capture_flag = False

        # Set the GUI text to back one
        self.ui.pushButton_5.setText("Start Capture Face")
        self.ui.pushButton_6.setText("Start Camera")

    def reset_windows(self):
        self.ui.label_3.setStyleSheet("QFrame{\n"
                                      "background:none;\n"
                                      "background-color: rgb(255, 255, 255);\n"
                                      "border:2px solid #4161AD;\n"
                                      "border-radius:5px\n"
                                      "}\n"
                                      "")
        self.ui.label_7.setStyleSheet("QFrame{\n"
                                      "background:none;\n"
                                      "background-color: rgb(255, 255, 255);\n"
                                      "border:2px solid #4161AD;\n"
                                      "border-radius:5px\n"
                                      "}\n"
                                      "")
        _translate = QtCore.QCoreApplication.translate
        self.ui.label_3.setText(_translate("MainWindow",
                                           "<html><head/><body><p><span style=\" font-size:14pt; "
                                           "font-weight:600;\">Image Frame</span></p></body></html>"))
        self.ui.label_7.setText(_translate("MainWindow",
                                           "<html><head/><body><p><span style=\" font-size:14pt; "
                                           "font-weight:600;\">Tracking Frame</span></p></body></html>"))

    def get_embedding(self, face_pixels):
        # scale pixel values
        face_pixels = face_pixels.astype('float32')
        # standardize pixel values across channels (global)
        mean, std = face_pixels.mean(), face_pixels.std()
        face_pixels = (face_pixels - mean) / std
        # transform face into one sample
        samples = expand_dims(face_pixels, axis=0)
        # make prediction to get embedding
        yhat = self.emb_model.predict(samples)
        return yhat[0]

    def start_training(self):

        # check for the person quantity
        if len(os.listdir(self.images_folder)) < 2:
            self.warning_obj.label_3.setText("At least two subjects are required.....")
            self.warning_window.show()
            return True

        name_dict = {}

        # Loading the images from the folder
        for n, img_folders in enumerate(os.listdir(self.images_folder)):
            name_dict[str(n)] = img_folders
            for img_path in os.listdir(os.path.join(self.images_folder, img_folders)):
                # Load the image
                image = cv2.imread(os.path.join(self.images_folder, img_folders, img_path))
                # Resizing the image
                image = cv2.resize(image, (self.image_size, self.image_size))
                # Normalizing
                self.embeddings.append(self.get_embedding(image))
                # Saving the image
                self.images.append(image)
                # Saving the label
                self.classes.append(n)

        # Save the dictionary as a dataframe
        result = pd.DataFrame(name_dict, index=[0])
        result.to_csv("names.csv", index=False)

        # Training over embeddings
        self.model.fit(self.embeddings, self.classes)

        return False

    def start_tracking_on_camera(self):

        # Current tracking report Dataframe
        current_report = pd.DataFrame(columns=['ID', 'Name', 'Date', 'Time'])

        if self.ui.pushButton_8.text() == "Start Tracking":

            self.ui.pushButton_8.setText("Loading training images")
            # Start training
            f = self.start_training()
            if f:
                return

            # Change the camera
            self.ui.pushButton_8.setText("Stop Tracking")

            # Camera stream Flag on
            self.camera_stream = True

            # Video capture from the camera
            self.video = cv2.VideoCapture(self.camera_id)

            # To calculate the frame rate
            fps_start_time = datetime.now()
            total_frames = 0

            while True:

                # Get the frame from camera
                ret, frame = self.video.read()

                # Load the names
                names_dict = pd.read_csv("names.csv").values[0]

                # To calculate the fps rate in the video from camera
                total_frames = total_frames + 1
                fps_end_time = datetime.now()
                time_diff = fps_end_time - fps_start_time
                if time_diff.seconds == 0:
                    fps = 0.0
                else:
                    fps = (total_frames / time_diff.seconds)

                fps_text = "FPS: {:.2f}".format(fps)

                # if frame is read
                if ret:
                    # convert the frame into grayscale and get the face Coordinates
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    faces = self.detector(gray)

                    # Check face or not found
                    if len(faces) != 0:

                        # Now get all the face in the frame
                        for face in faces:
                            x1 = face.left()
                            y1 = face.top()
                            x2 = face.right()
                            y2 = face.bottom()

                            # Now get the reign of interest of the face and get the prediction over that face
                            roi = frame[y1:y2, x1:x2]

                            # Images resizing and preprocessing
                            try:
                                roi = cv2.resize(roi, (self.image_size, self.image_size))
                            except:
                                continue

                            emb = self.get_embedding(roi)
                            pred = self.model.predict([emb])[0]

                            # Draw a green box over the face
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            cv2.rectangle(frame, (x2 - (x2 - x1), y2), (x2, y2 + 50), (0, 255, 0), -1)

                            # Display the id text
                            cv2.putText(frame, str(names_dict[pred]), (x1, y2 + 30), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                        (255, 255, 255), 2)

                            # Save the date and time of the person seen at last
                            ts = time.time()
                            date = datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
                            timeStamp = datetime.fromtimestamp(ts).strftime('%H:%M:%S')
                            current_report.loc[len(current_report)] = [pred, names_dict[pred], date, timeStamp]

                    # Write the count of the images
                    cv2.putText(frame, fps_text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                    # Show the image
                    result = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    height, width, channel = result.shape
                    step = channel * width
                    qImg = QImage(result.data, width, height, step, QImage.Format_RGB888)
                    self.ui.label_7.setPixmap(QtGui.QPixmap(qImg))

                    cv2.waitKey(1)

                    # Camera stream check
                    if not self.camera_stream:
                        break
        else:
            self.ui.pushButton_8.setText("Start Tracking")
            self.camera_stream = False

        # Drop the duplicate records
        current_report = current_report.drop_duplicates(subset=['ID'], keep='first')

        # Save the report
        self.report = pd.concat([self.report, current_report], axis=0)
        self.report.to_csv('report.csv', index=False)

        # Release the camera and destroy the windows
        self.video.release()

        # resetting the windowa
        self.reset_windows()

        # Reset the button text previous
        self.ui.pushButton_8.setText("Start Tracking")

    def view_Report(self):
        if not os.path.exists("report.csv"):
            return

        # Empty the table first
        for i in range(self.ui.table.rowCount()):
            self.ui.table.removeRow(i)

        # Load and add the csv file data into table widget
        with open("report.csv") as f:
            file_data = []
            row = csv.reader(f)

            for x in row:
                file_data.append(x)

            self.ui.table.setRowCount(0)
            file_data = iter(file_data)
            next(file_data)

            for row, rd in enumerate(file_data):
                self.ui.table.insertRow(row)
                for col, data in enumerate(rd):
                    self.ui.table.setItem(row, col, QtWidgets.QTableWidgetItem(str(data)))

        # Navigate to the Last Seen Log Window
        self.ui.stackedWidget.setCurrentWidget(self.ui.page_5)


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    main = Main()
    main.main_window.show()
    sys.exit(app.exec_())
