import os
import io
import tkinter as tk
from tkinter import filedialog
from PIL import ImageTk, Image
import matplotlib.pyplot as plt
import sam2_repository
import numpy as np
import cv2
import image_processor
from object_prompt import ObjectPrompt
from object_manager import ObjectManager
import colour_map
from threading import Thread
from time import sleep


class DataAnalysisToolGUI:
   
    def __init__(self, master):
        self.master = master
        self.master.title("DataAnalysisTool")

        # Create a frame for the GUI and center it
        self.frame = tk.Frame(self.master)
        self.frame.pack(expand=True, padx=10, pady=10)
        self.frame.grid_rowconfigure(0,  weight=1)
        self.frame.grid_columnconfigure(0, weight=1)

        # Create a border for the Controls
        self.border = tk.Frame(self.frame, borderwidth=2, relief="groove")
        self.border.grid(column=0, row=0, sticky="nsew")

        # Create a border for the input ImageFrame
        self.image_border = tk.Frame(self.frame, borderwidth=2, relief="groove")
        self.image_border.grid(column=1, row=0, sticky="nsew")

        # Create a label to display the input image
        self.input_frame = tk.Label(self.image_border)
        self.input_frame.grid(column = 1, row = 1, sticky="nsew", padx = 10, pady = 10)
        self.input_frame.bind("<ButtonRelease-1>", self.click_event)

        # Create a border for the input ImageFrame
        # self.output_image_border = tk.Frame(self.frame, borderwidth=2, relief="groove")
        # self.output_image_border.grid(column=1, row=1, sticky="nsew")

        # Create a label to display the output image
        self.output_frame = tk.Label(self.image_border)
        self.output_frame.grid(column = 1, row = 2, sticky="nsew", padx = 10, pady = 10)

        # Create a "Load Image Directory" button
        self.load_button = tk.Button(self.border, text="Load Directory", command=self.load_directory)
        self.load_button.grid(column = 1, row = 1, padx = 10, pady = 10, sticky="w")

        # Create a Label for channel OptionMenu
        self.annotation_label = tk.Label(self.border, text=' Label ', relief=tk.FLAT)
        self.annotation_label.grid(column = 1, row = 2, padx = 10, pady = 10, sticky="sw")

        # Object Manager
        self.obj_mnger = ObjectManager()
        self.label_lists = self.obj_mnger.get_object_lists()

        # Create a Option menu for setting channel
        self.label_var = tk.StringVar()
        self.label_var.set(self.label_lists[0])
        self.label_option = tk.OptionMenu(self.border, self.label_var, *self.label_lists)
        self.label_option.config(width=13)
        self.label_option.grid(column = 1, row = 3, padx = 10, pady = 10, sticky="nsew")

        # Create a "set" button
        self.set_label_button = tk.Button(self.border, text=" Set ", command=self.set_label)
        self.set_label_button.grid(column = 1, row = 6, padx = 10, pady = 10, sticky="nw")

        # Create a "clear" button
        self.clear_label_button = tk.Button(self.border, text="Clear", command=self.clear_label)
        self.clear_label_button.grid(column = 1, row = 6, padx = 10, pady = 10, sticky="ne")

        # Create a Label for channel OptionMenu
        self.objects_label = tk.Label(self.border, text=' Tracking Objects ', relief=tk.FLAT)
        self.objects_label.grid(column = 1, row = 7, padx = 10, pady = 10, sticky="sw")

        # Create a Option menu for Tracking Objects
        self.objects_var = tk.StringVar()
        self.objects_var.set('object_0')
        self.objects_option = tk.OptionMenu(self.border, self.objects_var, "object_0", command=self.object_select)
        self.objects_option.config(width=10)
        self.objects_option.grid(column = 1, row = 8, padx = 10, pady = 10, sticky="nsew")

    
        # Set Obj instances
        self.object_prompts = {}
        obj_promt = ObjectPrompt(0)
        self.object_prompts.update({0: obj_promt})
        self.current_object = self.object_prompts.get(0)
        self.objects_option_len = 1

        # Create a "Add" button
        self.object_add_button = tk.Button(self.border, text="  Add ", command=self.add_object)
        self.object_add_button.grid(column = 1, row = 9, padx = 10, pady = 10, sticky="nw")

        # Create a "Remove" button
        self.object_remove_button = tk.Button(self.border, text="Remove", command=self.remove_object)
        self.object_remove_button.grid(column = 1, row = 9, padx = 10, pady = 10, sticky="ne")

        # Set object output image
        image = np.zeros((image_processor.h, image_processor.w, 3), dtype = np.uint8)
        image = 255*image
        self.output_image = Image.fromarray(image)
        
        # Set object input mode
        self.object_input_mode = 'none'

        # Create a "Preview" button
        self.preview_button = tk.Button(self.border, text="Preview", command=self.check_preview)
        self.preview_button.grid(column = 1, row = 10, padx = 10, pady = 10, sticky="nw")

        # Create a "start tracking" button
        self.start_tracking_button = tk.Button(self.border, text="Start Tracking", command=self.start_tracking)
        self.start_tracking_button.grid(column = 1, row = 11, padx = 10, pady = 10, sticky="nw")

        # Set Tracking Done
        self.tracking_done = False

        # Create a "start over" button
        self.start_over_button = tk.Button(self.border, text="Start Over", command=self.start_over)
        self.start_over_button.grid(column = 1, row = 12, padx = 10, pady = 10, sticky="nw")

        # Create a "save data" button
        self.save_data_button = tk.Button(self.border, text="Save Data", command=self.save_data)
        self.save_data_button.grid(column = 1, row = 13, padx = 10, pady = 10, sticky="nw")

        # Player
        self.player_state = False
        self.player_thread = None

        sam2_repository.test()


    
    def load_directory(self):
        # Open a file selection dialog box to choose an image file
        self.file_path = filedialog.askdirectory(title="Select Input Folder")
        print(self.file_path)
        num_images = image_processor.image_preprocessing(self.file_path)
        self.frame_num = num_images
        self.showFrameController()
        self.showImage(0)
        self.reset()


    def load_image(self):
        # Open a file selection dialog box to choose an image file
        self.file_path = filedialog.askopenfilename(title="Select Image File", filetypes=[('JPG Files', '*.jpg'), ('Png Files', '*.png'), ('jpeg Files', '*.jpeg'), ('bmp Files', '*.bmp'), ('gif Files', '*.gif')])
        print(self.file_path)
        self.origin_image = Image.open(self.file_path)
        photo = ImageTk.PhotoImage(self.origin_image)
        self.input_frame.configure(image=photo)
        self.input_frame.image = photo

        # self.output_frame.configure(image=photo)
        # self.output_frame.image = photo
        # self.showSliceIDBar()

    def imagePreProcessing(self):
        entries = os.listdir(self.file_path+'/')
        entries = [x for x in entries if x.endswith('.png')]
        print(entries)
        print(len(entries))

    
    def showFrameController(self):
        # Create a "play" button
        self.set_label_button = tk.Button(self.image_border, text="play", command=self.play_frame)
        self.set_label_button.grid(column = 1, row = 3, padx = 10, pady = 10, sticky="nw")

        # Create a "pause" button
        self.set_label_button = tk.Button(self.image_border, text="pause", command=self.pause_frame)
        self.set_label_button.grid(column = 1, row = 3, padx = 70, pady = 10, sticky="nw")

        #  Create a Scale widget for setting Slice ID        
        self.slice_var = tk.IntVar()
        self.slice_var.set(0)
        self.slice_scale = tk.Scale(self.image_border, width=20, length = image_processor.w, from_=0, to=self.frame_num - 1, orient=tk.HORIZONTAL, label="Frame ID", variable=self.slice_var)
        self.slice_scale.bind("<ButtonRelease-1>", self.updateFrameId)
        self.slice_scale.grid(column = 1, row = 4, sticky="sw", padx = 10, pady = 10)

    def showImage(self, frame_id):
        file_name = '{}.jpg'.format(frame_id)
        full_path = image_processor.image_preprocessing_output + '/' + file_name
        self.origin_image = Image.open(full_path)
        self.updateImage(self.origin_image)
        if self.tracking_done:
            full_path = 'output/{}.png'.format(frame_id)
            output_image = Image.open(full_path)
        else:
            output_image = self.output_image
        self.updateOutputImage(output_image)

    def updateOutputImage(self, update_image):
        # Resize the image to fit in the image_label label
        width, height = update_image.size
        print(width, height)
        photo = ImageTk.PhotoImage(update_image)
        self.output_frame.configure(image=photo)
        self.output_frame.image = photo
        

    def updateImage(self, update_image):
        # Resize the image to fit in the image_label label
        width, height = update_image.size
        print(width, height)
        photo = ImageTk.PhotoImage(update_image)
        self.input_frame.configure(image=photo)
        self.input_frame.image = photo

    def updateFrameId(self, event):
        evt_name = str(event)
        print(evt_name)
        frame_id = self.slice_var.get()
        print('frame_id: ', frame_id)
        self.showImage(frame_id)
        for obj in self.object_prompts.values():
            obj.clear()

    def object_select(self, event):
        print(event)
        obj_idx = self.objects_option["menu"].index(self.objects_var.get())
        print('current index: ' , obj_idx)
        self.current_object = self.object_prompts.get(obj_idx)


    def click_event(self, event):
        evt_name = str(event)
        print(evt_name, self.object_input_mode)
        if (self.object_input_mode == 'none'):
            return
        
        if self.object_input_mode == 'add':
            self.current_object.addPrompt([event.x,event.y], 1)
        else:
            self.current_object.addPrompt([event.x,event.y], 0)
        print('positions:', str(self.current_object.input_position))
        print('label:', str(self.current_object.input_label))


        np_origin = np.array(self.origin_image)
        for obj_prompt in self.object_prompts.values():
            if not obj_prompt.isActivate():
                continue
            obj_id = obj_prompt.getId()
            for i in range(0, len(obj_prompt.input_position)):
                pos = obj_prompt.input_position[i]
                label = obj_prompt.input_label[i]
                if label == 1:
                    cv2.drawMarker(np_origin, (pos[0], pos[1]), colour_map.colour_map(obj_id), cv2.MARKER_STAR, 10, 1)
                if label == 0:
                    cv2.drawMarker(np_origin, (pos[0], pos[1]), (250, 0, 0), cv2.MARKER_STAR, 10, 1)
        update_image = Image.fromarray(np_origin)
        self.updateImage(update_image)
        # update object options if needed 
        self.update_options()

    def update_options(self):
        if (self.current_object.getId() != self.objects_option_len -1):
            print('update_options return')
            return
        #self.objects_var.set('')
        choice = 'object_{}'.format(self.objects_option_len)
        self.objects_option['menu'].add_command(label = choice, command=tk._setit(self.objects_var, choice, self.object_select))
        # update object_prompts
        obj_promt = ObjectPrompt(self.objects_option_len)
        self.object_prompts.update({self.objects_option_len: obj_promt})
        self.objects_option_len += 1


    def play_frame(self):
        player_thread = Thread(target=self.start_playback)
        self.player_state = True
        player_thread.start()

    def start_playback(self):
        frame_id = self.slice_var.get()
        for i in range(frame_id, self.frame_num):
            # print(self.player_state)
            if self.player_state == False:
                break
            self.showImage(i)
            self.slice_var.set(i)
            sleep(0.066)
        

    def pause_frame(self):
        self.player_state = False
        # self.player_thread.join()
        # self.player_thread = None

    def set_label(self):
        entity_name = self.label_var.get()
        obj_id = self.current_object.getId()
        print("entity_name: {}, obj_id: {}".format(entity_name, obj_id))
        self.obj_mnger.set(obj_index= obj_id, obj_name=entity_name)

    def clear_label(self):
        obj_id = self.current_object.getId()
        self.obj_mnger.unset(obj_id)

    def add_object(self):
        self.object_input_mode = 'add'

    def remove_object(self):
        self.object_input_mode = 'remove'

    def check_preview(self):
        frame_id = self.slice_var.get()
        update_image, _, _, h, w = sam2_repository.doImagePredic(image_processor.image_preprocessing_output, frame_id, self.object_prompts, self.obj_mnger)
        #update_image = Image.open('output/{}.png'.format(frame_id))
        self.updateOutputImage(update_image)
    
    def start_tracking(self):
        self.tracking_done = True
        frame_id = self.slice_var.get()
        print('frame_id: ', frame_id)
        self.showImage(frame_id)
        return
        frame_id = self.slice_var.get()
        update_image, predictor, inference_state, h, w = sam2_repository.doImagePredic(image_processor.image_preprocessing_output, frame_id, self.object_prompts, self.obj_mnger)
        self.tracking_done = sam2_repository.doVideoPredic(predictor, inference_state, self.frame_num, h, w, objMngr = self.obj_mnger)

        self.updateOutputImage(update_image)
        
    def reset(self):
        self.tracking_done = False
        self.showImage(0)
        self.slice_var.set(0)
        self.label_var.set(self.label_lists[0])
        self.objects_var.set('object_0')
        self.object_prompts.clear()
        
        obj_promt = ObjectPrompt(0)
        self.object_prompts.update({0: obj_promt})
        self.current_object = self.object_prompts.get(0)
        self.objects_option_len = 1

        self.objects_option['menu'].delete(1, 'end')

    
    def start_over(self):
        print('Start Over')
        self.reset()

    def save_data(self):
        pass

            
if __name__ == "__main__":
    root = tk.Tk()
    gui = DataAnalysisToolGUI(root)
    root.mainloop()