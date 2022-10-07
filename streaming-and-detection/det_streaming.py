from os.path import join
import argparse
from pickletools import uint8
import gi
from matplotlib import image
from numpy import ndarray
gi.require_version('Gtk', '3.0')
from gi.repository import Gtk, Gdk, GdkPixbuf, GLib
import threading
import time
import socket,os,struct

#save image and visualize
import numpy as np
import cv2 
import binascii 
import io 
from PIL import Image

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
#from detectron2.utils.analysis import parameter_count, parameter_count_table
#from detectron2.modeling.meta_arch.build import build_model

VERBOSE=False
RECEIVE_TIMESTAMP=False #only used for the dataset framework collector

deck_ip = None
deck_port = None

cfg = get_cfg()
# add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
# find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
predictor = DefaultPredictor(cfg)

# view model characteristics
#model = build_model(cfg) 
#tab = parameter_count_table(model, 2) 
#print(tab)

def save_image(imgdata, number_of_images):
    decoded = cv2.imdecode(np.frombuffer(imgdata, np.uint8), -1)
    image_name = str(number_of_images)+".jpg"
    try: cv2.imwrite(join(images_folder_path,image_name), decoded)
    except: print('couldnt decode image, data lenght was', len(imgdata))
    

class ImgThread(threading.Thread):
    def __init__(self, callback):
        threading.Thread.__init__(self, daemon=True)
        self._callback = callback

    def run(self):
        print("Connecting to socket on {}:{}...".format(deck_ip, deck_port))
        client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client_socket.connect((deck_ip, deck_port))
        print("Socket connected")

        imgdata = None
        imgdata_complete = None
        number_of_images = 0
        img = None
        

        while(1):
            strng = client_socket.recv(512)
            if VERBOSE: print("\nsize of packet received:", len(strng), "\n")
            if VERBOSE: print (binascii.hexlify(strng))

            # Look for start-of-frame and end-of-frame
            start_idx = strng.find(b"\xff\xd8")
            end_idx = strng.find(b"\xff\xd9")

            # Concatenate image data, once finished send it to the UI
            if start_idx >= 0:
                number_of_images+=1
                #append end of packet
                imgdata += strng[:start_idx]
                #put in another variable the complete image
                imgdata_complete = imgdata

                #start the acquisition of the new image
                imgdata = strng[start_idx:]

                # search for the footer in the complete_image and ignore it (Temporal fix: the footer is transmitted not at the end of each image so we just discard it to not break the image)
                end_idx = imgdata_complete.find(b"\xff\xd9")
                if end_idx >= 0 and imgdata_complete:
                    imgdata_complete = imgdata_complete[0:end_idx] + imgdata_complete[end_idx+2:]
                
                if RECEIVE_TIMESTAMP: # remove last 8 bytes, which are just timestamp
                    timestamp = imgdata_complete[-8:]
                    imgdata_complete = imgdata_complete[:-8]

                # Now append the jpeg footer at the end of the complete image. We do this before saving or visualizing the image, so it can be decoded correctly
                imgdata_complete = imgdata_complete + (b"\xff\xd9")

                if VERBOSE: print('len strng %d  \t Bytes imgdata  %d\t \n\n' % (len(strng), len(imgdata_complete)-308 )) #308 = len(header)+len(footer)

                if SAVE_IMAGES==True and (number_of_images % 5 ==0 ): #saves just one every 5 images to not overload
                    save_image(imgdata_complete, number_of_images)
                    
                
                if (number_of_images % 30 ==0 ): # uses one image out of 30

                    im = cv2.imdecode(np.frombuffer(imgdata_complete, np.uint8), -1) 
                    if type(im) == type(None):
                        print("decoding not success")
                        continue

                    im = cv2.cvtColor(im,cv2.COLOR_BGR2RGB)

                    outputs = predictor(im)
                    v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
                    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))

                    im_out = out.get_image()[:, :, ::-1]
                    result = cv2.imwrite("debug_out.jpg",im_out) # saves the image with the prediction
                    with open("debug_out.jpg", 'rb') as f:
                        byte_im = f.read() # image with the prediction in the byte array format

                    try: #show frame
                        self._callback(byte_im)
                    except gi.repository.GLib.Error:
                        print ("image not shown")
                        pass

            else: # Continue receiving the image
                if imgdata==None:
                    imgdata=strng
                else:
                    imgdata += strng

       
          
# UI for showing frames from AI-deck example
class FrameViewer(Gtk.Window):

    def __init__(self):
        super(FrameViewer, self).__init__()
        self.frame = None
        self.init_ui()
        self._start = None
        self.set_default_size(374, 294)

    def init_ui(self):            
        self.override_background_color(Gtk.StateType.NORMAL, Gdk.RGBA(0, 0, 0, 1))
        self.set_border_width(20)
        self.set_title("Connecting...")
        self.frame = Gtk.Image()
        f = Gtk.Fixed()
        f.put(self.frame, 10, 10)
        self.add(f)
        self.connect("destroy", Gtk.main_quit)
        self._thread = ImgThread(self._showframe)
        self._thread.start()

    def _update_image(self, pix):
        self.frame.set_from_pixbuf(pix)

    def _showframe(self, imgdata):
        # Add FPS/img size to window title
        if (self._start != None):
            fps = 1 / (time.time() - self._start)
            GLib.idle_add(self.set_title, "{:.1f} fps / {:.1f} kb".format(fps, len(imgdata)/1000))
        self._start = time.time()
        img_loader = GdkPixbuf.PixbufLoader()

        # Try to decode JPEG from the data sent from the stream
        try:
            img_loader.write(imgdata)
            pix = img_loader.get_pixbuf()
            GLib.idle_add(self._update_image, pix)
        except gi.repository.GLib.Error:
            print("Could not set image!")
        img_loader.close()

# Args for setting IP/port of AI-deck. Default settings are for when
# AI-deck is in AP mode.
parser = argparse.ArgumentParser(description='Connect to AI-deck JPEG streamer example')
parser.add_argument("-n",  default="192.168.4.1", metavar="ip", help="AI-deck IP")
parser.add_argument("-p", type=int, default='5000', metavar="port", help="AI-deck port")
parser.add_argument('--save_images', help='save images on your pc', action='store_true')
parser.add_argument('--save_images_path', help='folder where images are saved', default='/home/llamberti/work/dataset_save_folder/')
args = parser.parse_args()


SAVE_IMAGES = args.save_images
images_folder_path = args.save_images_path
deck_port = args.p
deck_ip = args.n

fw = FrameViewer()
fw.show_all()
Gtk.main()