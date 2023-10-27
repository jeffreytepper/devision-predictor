import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk, ImageFilter

import os

from tifffile import imread

from csbdeep.utils import normalize
from csbdeep.io import save_tiff_imagej_compatible

from stardist.models import StarDist2D

class Slideshow():

    def __init__(self, app, root):

        self.app = app

        self.current_index = -1
        self.images = []

        self.title_label = tk.Label(root)
        self.title_label.grid(row=1, column=0, columnspan=2)

        left_img_frame = tk.Frame(root)
        left_img_frame.grid(row=2, column=0, padx=10, pady=5)
        self.test_image_label = tk.Label(left_img_frame)
        self.test_image_label.pack()

        right_img_frame = tk.Frame(root)
        right_img_frame.grid(row=2, column=1, padx=10, pady=5)
        self.prediction_image_label = tk.Label(right_img_frame)
        self.prediction_image_label.pack()

        self.numitems_label = tk.Label(root)
        self.numitems_label.grid(row=3, column=0, columnspan=2)

        button_frame = tk.Frame(root, pady=10)
        button_frame.grid(row=4, column=0, columnspan=2)

        next_button = tk.Button(button_frame, text="Next", command=self.next_image)
        prev_button = tk.Button(button_frame, text="Prev", command=self.prev_image)
        next_button.pack(side='right')
        prev_button.pack(side='left')

    def append_image(self, imagepath):
        image = Image.open(imagepath)
        image = image.resize((300,300))
        tk_image = ImageTk.PhotoImage(image)
        self.images.append([tk_image,None,-1])

        self.next_image()

    def add_prediction(self, imagepath):
        image = Image.open(imagepath)
        image = image.resize((300,300))
        tk_image = ImageTk.PhotoImage(image)
        self.images[self.current_index][1] = tk_image

        self.prediction_image_label.config(image=tk_image)

    def to_index(self, index):
        self.current_index = index % len(self.images)

        self.title_label.config(text=os.path.basename(self.app.images[self.current_index]))
        test_image = self.images[self.current_index][0]
        prediction_image = self.images[self.current_index][1]
        num_items = self.images[self.current_index][2]

        self.test_image_label.config(image=test_image)

        if prediction_image:
            self.prediction_image_label.config(image=prediction_image)
            self.numitems_label.config(text='Number of Items:' + str(num_items))
        else:
            self.prediction_image_label.config(image='')
            self.numitems_label.config(text='')

    def next_image(self):
        self.to_index(self.current_index + 1)

    def prev_image(self):
        self.to_index(self.current_index - 1)



class App():

    def __init__(self):
        self.root = tk.Tk()
        self.images = []
        self.model = StarDist2D.from_pretrained('2D_demo')
        self.output_dir = os.getcwd()

        header_frame = tk.Frame(self.root, pady=10)
        header_frame.grid(row=0, column=0, columnspan=2)

        open_image_button = tk.Button(header_frame, text="Select Images", command=self.select_images)
        set_model_button = tk.Button(header_frame, text="Select Model", command=self.set_model)
        set_output_dir = tk.Button(header_frame, text="Select Output Location", command=self.select_output_dir)
        open_image_button.pack(side='left')
        set_model_button.pack(side='right')
        set_output_dir.pack(side='right')

        self.slideshow = Slideshow(self, self.root)
        
        bottom_frame = tk.Frame(self.root, pady=10)
        bottom_frame.grid(row=5, column=0, columnspan=2)

        predict_button = tk.Button(bottom_frame, text="Predict", command=self.predict)
        predict_all_button = tk.Button(bottom_frame, text="Predict All", command=self.predict_all)
        predict_button.pack()
        predict_all_button.pack()

    def add_image(self, imagepath):
        self.images.append(imagepath)
        self.slideshow.append_image(imagepath)
    
    def set_model(self):
        dirpath = filedialog.askdirectory(initialdir='.')
        self.model = StarDist2D(None, name='stardist', basedir=dirpath)
            
    def select_images(self):
        filepaths = filedialog.askopenfilenames(initialdir='.')
        for path in filepaths:
            self.add_image(path)

    def select_output_dir(self):
        self.output_dir = filedialog.askdirectory(initialdir='.')


    def predict_all(self):
        self.slideshow.to_index(0)
        output_file = open(os.path.join(self.output_dir, 'num_items.csv'), 'w')

        for imagepath in self.images:
            labels, details = self.predict()

            num_items = len(details['points'])
            title = os.path.basename(self.images[self.slideshow.current_index])
            output_file.write(title+','+str(num_items)+'\n')

            self.slideshow.next_image()

        output_file.close()
            
            

    def predict(self):
        imagepath = self.images[self.slideshow.current_index]
        img = imread(imagepath)
        img = normalize(img, 1, 99.8, axis=(0,1))
        labels, details = self.model.predict_instances(img)

        save_filedir = os.path.join(self.output_dir, 'labels')

        if not os.path.exists(save_filedir):
            os.makedirs(save_filedir)

        save_filepath = os.path.join(save_filedir, os.path.basename(imagepath))

        save_tiff_imagej_compatible(save_filepath, labels, axes='YX')
        self.slideshow.add_prediction(save_filepath)

        num_items = len(details['points'])
        self.slideshow.images[self.slideshow.current_index][2] = num_items
        self.slideshow.numitems_label.config(text='Number of Items:' + str(num_items))

        return labels, details
    

app = App()
root = app.root

root.title("Devision Predictor")

root.mainloop()