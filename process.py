import os
import sys
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
import numpy as np
import yaml

from models.common import DetectMultiBackend
from utils.datasets import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from utils.general import (LOGGER, check_file, check_img_size, check_imshow, check_requirements, colorstr,
                           increment_path, non_max_suppression, print_args, scale_coords, strip_optimizer, xyxy2xywh)
from utils.torch_utils import select_device, time_sync
import random
import pandas as pd
import shutil

@torch.no_grad()
def load_yolo_predictor(device=0, 
                        dnn=False, 
                        data="",
                        weights=[]):
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data)
    return model

@torch.no_grad()
def run(
       im,
       model,
       imgsz=(1024, 1024), 
       conf_thres=0.05,
       iou_thres=0.2,
       max_det=1000,
       classes=None,
       agnostic_nms=False,
       augment=False,
       half=False,
       device=0,
       visualize=False,
       spacing=[1, 1],
       image_index=0):
    
    stride, names, pt, jit, onnx, engine = model.stride, model.names, model.pt, model.jit, model.onnx, model.engine
    imgsz = check_img_size(imgsz, s=stride)  # check image size
    bs = 1  # batch_size
    model.warmup(imgsz=(1, 3, *imgsz), half=half)  # warmup
    dt, seen = [0.0, 0.0, 0.0], 0
    
    im = np.transpose(im, (2, 0, 1))
    im = torch.from_numpy(im).to(device)
    im = im.half() if half else im.float()  # uint8 to fp16/32
    im /= 255  # 0 - 255 to 0.0 - 1.0
    if len(im.shape) == 3:
        im = im[None]  # expand for batch dim
    pred = model(im, augment=augment, visualize=visualize)
    
    scores = pred[:, :, 4]
    if torch.max(scores) >= 0.4:
        conf_thres = 0.10
    else:
        conf_thres = 0.05  
    if torch.max(scores) >= 0.5:
        conf_thres = 0.15
    else:
        conf_thres = 0.05  
    if torch.max(scores) >= 0.7:
        conf_thres = 0.25
    else:
        conf_thres = 0.05 
    if torch.max(scores) <= 0.2:
        pred[:, :, 4] = pred[:, :, 4] / 2
        
    pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
    boxes = []
    x_y_spacing = [spacing[0], spacing[1], spacing[0], spacing[1]]
    for i, det in enumerate(pred):  # per image
        if len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(im.shape[2:], det[:, :4], (1024, 1024)).round()
            for *xyxy, conf, cls in reversed(det):
                box = {}
                box['corners'] = []
                bbox = [xyxy[0].item(), xyxy[1].item(), xyxy[2].item(), xyxy[3].item()]
                bbox = [bbox[0] * x_y_spacing[0], bbox[1] * x_y_spacing[1],
                       bbox[2] * x_y_spacing[2], bbox[3] * x_y_spacing[3]]
                confidence = conf.item()
                
                if confidence >= 0.8:
                    confidence = 0.95
                    
                bbox = [np.round(bbox[0], 2), np.round(bbox[1], 2), 
                        np.round(bbox[2], 2), np.round(bbox[3], 2)]
                bottom_left = [bbox[0], bbox[1], image_index]
                bottom_right = [bbox[2], bbox[1], image_index]
                top_left = [bbox[0], bbox[3], image_index]
                top_right = [bbox[2], bbox[3], image_index]
                box['corners'].extend([top_right, top_left, bottom_left, bottom_right])
                box['probability'] = np.round(confidence, 2)
                boxes.append(box)
    return boxes

import os
import numpy as np
import SimpleITK as sitk

def convert_file(mri_file_path, png_file_path, auto_contrast=False):
    # NB: only one png output file at a time, so not good if len(mri_file_path)>1

    # Check the existence of the MRI file
    if len(mri_file_path) != 0:
        if type(mri_file_path[0]) == str:
            if not os.path.exists(mri_file_path[0]):
                raise Exception('Source file "%s" does not exists' % mri_file_path)

    # Remove the output png, if non existent
    flag_conversion = dicom2image(mri_file_path[0], png_file_path, auto_contrast)
    return flag_conversion


def dicom2image(input_file_name, output_file_name, auto_contrast=False):
    # Initialize a file reader
    if type(input_file_name) == str:
        image = sitk.ReadImage(input_file_name)
    else:
        image = input_file_name
    spacing = image.GetSpacing()
    # Ensure that, if the image is greyscale, it has only 2 dimensions
    
    # If greyscale rescale the image between 0 and 255 and save it as a Uint8
    if image.GetNumberOfComponentsPerPixel() == 1:
        image = sitk.RescaleIntensity(image, 0, 255)
        image = sitk.Cast(image, sitk.sitkUInt8)

    # Get a numpy array from the image
    npa = sitk.GetArrayFromImage(image)

    # Perform contrast enhancement(numpy array required)
    npa = auto_contrast_enhancement(npa)

    # If required, perform contrast enhancement
    if auto_contrast:
        image = sitk.GetImageFromArray(npa)
        image = sitk.Cast(image, sitk.sitkUInt8)

    # Write the image into the output file of iterest
    return image, spacing


def auto_contrast_enhancement(img, min_freq=0.001, null_perc=0.5):
    # The image is initially normalized between 0 and 255
    p_low = np.min(img)
    p_high = np.max(img)
    img = (img.astype(float) - p_low) / (p_high - p_low) * 255

    # The histogram of the normalized image is computed
    hist, bins = np.histogram(img.ravel(), bins=np.arange(0, 256))

    # Histogram information is used to automatically identify a lower and upper treshold for the windowing of the grey values
    p_low, p_high = optimize_gray_range(hist, bins, min_freq, null_perc)

    # The windowing of the grey level is performed
    img = np.maximum(img, p_low)
    img = np.minimum(img, p_high)

    # The selected portion of the histogram is mapped between 0 and 255
    img = (img.astype(float) - p_low) / (p_high - p_low) * 255

    return img


def optimize_gray_range(hist, bins, min_freq=0.001, null_perc=0.5):
    # Get the relative frequencies from the absolute frequencies of the histogram
    n = np.sum(hist)
    hist = hist / n

    # Get the intial number of "empty bins" (relative frequency below the threshold)
    non_empty_bins = bins[np.where(hist > min_freq)]
    n_empty_tot = len(non_empty_bins)

    # Shrink the histogram until the empty bins only become a small percentage of the initial number
    n_empty = n_empty_tot
    while n_empty / n_empty_tot > null_perc:
        non_empty_bins = non_empty_bins[1:-1]
        hist_new = hist[non_empty_bins[0]:non_empty_bins[-1]]
        n_empty = len(np.where(hist_new > min_freq))

    # The lower and upper end of the histogram are the extremes of the non_empty_bins variable

    low_ref = non_empty_bins[0]
    high_ref = non_empty_bins[-1]

    return low_ref, high_ref

def convert_nha_to_png(input_dir, output_dir):
    for fname in os.listdir(input_dir):
        mri_file_path = os.path.join(input_dir, fname)
        if fname.replace(".mha", ".png") not in os.listdir(output_dir):
            png_file_path = os.path.join(output_dir, fname.replace(".mha", ".png"))
            image, spacing = convert_file([mri_file_path], png_file_path, auto_contrast=True)
            sitk.WriteImage(image, png_file_path)
        

def make_yolo_format_files(input_dir, labels_yolo_dir):
    metadata_file = os.path.join(input_dir, 'metadata.csv')
    data = pd.read_csv(metadata_file)
    images_dir = os.path.join(input_dir, 'images')
    for fname in os.listdir(images_dir):
        mha_fname = fname.replace(".png", ".mha")
        df_in = data[data["img_name"] == mha_fname]
        rows = []
        for ix_row, row in df_in.iterrows():
            if row["label"] != 0:
                x_min, y_min, w, h = row["x"], row["y"], row["width"], row["height"]
                x_max, y_max = x_min + w, y_min + h
                x_c, y_c = (x_min + x_max) / 2, (y_min + y_max) / 2
                norm_x_c, norm_y_c = x_c / 1024, y_c / 1024
                norm_w, norm_h = w / 1024, h / 1024
                text = "0 " + str(norm_x_c) + " " + str(norm_y_c) + " " + str(norm_w) + " " + str(norm_h)
                rows.append(text)
        if len(rows) != 0:
            final_text = "\n".join(rows)
        else:
            final_text = ""
        label_file = os.path.join(labels_yolo_dir, fname.replace(".mha", ".txt"))
        with open(label_file, "w") as fhandle:
            fhandle.write(final_text)
    
    
def make_split_v1(input_dir, image_directory, yaml_dir):
    "It assumes that there are more clean images than images with nodules"
    percent_nodes = 0.85
    ratio_clean_nodules = 1.2
    splits = 3
    subsplits = 3
    metadata_file = os.path.join(input_dir, 'metadata.csv')
    data = pd.read_csv(label_file)
    groups = data.groupby(by="img_name")
    nodules = []
    cleans = []
    for key, group in groups:
        if group[group["label"] == 0].shape[0] == 1:
            cleans.append(key)
        else:
            nodules.append(key)
    nodule_paths = [os.path.join(image_directory, x.replace(".mha", ".png")) for x in nodules]
    clean_paths = [os.path.join(image_directory, x.replace(".mha", ".png")) for x in cleans]
    for split_ix in range(splits):
        nodule_split = [x for x in nodule_paths if random.uniform(0, 1) <= 0.85]
        length_nodules = len(nodule_split) 
        clean_split = [x for x in clean_paths if random.uniform(0, 1) <= 0.85] # 1.2 X nodule_numbers
        length_clean = len(clean_split)
        should_be_clean_length = len(nodule_split) * ratio_clean_nodules 
        for subsplit_ix in range(subsplits):
            clean_subsplit = random.sample(clean_split, should_be_clean_length)
            txt_filename = "train_V1_" + str(split_ix) + str(subsplit_ix) + ".txt"
            txt_filename = os.path.join(yaml_dir, txt_filename)
            text_train = "\n".join(nodule_split + clean_subsplit)
            with open(txt_filename, "w") as fhandle:
                fhandle.write(text_train)
            
            yaml_filename = os.path.join(yaml_dir, "nodule_V1_" + str(split_ix) + str(subsplit_ix) + ".yaml")
            data_yaml = dict(
                train = txt_filename ,
                val   = txt_filename,
                nc    = 1,
                names = ["nodule"]
                )
            with open(yaml_filename, 'w') as outfile:
                yaml.dump(data_yaml, outfile, default_flow_style=False)

def make_split_v2(input_dir, yaml_dir):
    "It assumes that there are more clean images than images with nodules"
    percent_nodes = 1.0
    ratio_clean_nodules = 1.2
    splits = 3
    subsplits = 3
    metadata_file = os.path.join(input_dir, 'metadata.csv')
    data = pd.read_csv(label_file)
    groups = data.groupby(by="img_name")
    nodules = []
    cleans = []
    for key, group in groups:
        if group[group["label"] == 0].shape[0] == 1:
            cleans.append(key)
        else:
            nodules.append(key)
    nodule_paths = [os.path.join(image_directory, x.replace(".mha", ".png")) for x in nodules]
    clean_paths = [os.path.join(image_directory, x.replace(".mha", ".png")) for x in cleans]
    for split_ix in range(splits):
        nodule_split = [x for x in nodule_paths if random.uniform(0, 1) <= 0.85]
        length_nodules = len(nodule_split) 
        clean_split = [x for x in clean_paths if random.uniform(0, 1) <= 0.85] # 1.2 X nodule_numbers
        length_clean = len(clean_split)
        should_be_clean_length = len(nodule_split) * ratio_clean_nodules
        for subsplit_ix in range(subsplits):
            clean_subsplit = random.sample(clean_split, should_be_clean_length)
            txt_filename = "train_V1_" + str(split_ix) + str(subsplit_ix) + ".txt"
            txt_filename = os.path.join(yaml_dir, txt_filename)
            text_train = "\n".join(nodule_split + clean_subsplit)
            with open(txt_filename, "w") as fhandle:
                fhandle.write(text_train)
            
            yaml_filename = os.path.join(yaml_dir, "nodule_V1_" + str(split_ix) + str(subsplit_ix) + ".yaml")
            data_yaml = dict(
                train = txt_filename ,
                val   = txt_filename,
                nc    = 1,
                names = ["nodule"]
                )
            with open(yaml_filename, 'w') as outfile:
                yaml.dump(data_yaml, outfile, default_flow_style=False)
        
def make_split_v3(input_dir, yaml_dir):
    "It assumes that there are more clean images than images with nodules"
    percent_nodes = 1.0
    ratio_clean_nodules = 1.2
    splits = 3
    subsplits = 3
    metadata_file = os.path.join(input_dir, 'metadata.csv')
    data = pd.read_csv(label_file)
    groups = data.groupby(by="img_name")
    nodules = []
    cleans = []
    for key, group in groups:
        if group[group["label"] == 0].shape[0] == 1:
            cleans.append(key)
        else:
            nodules.append(key)
    nodule_paths = [os.path.join(image_directory, x.replace(".mha", ".png")) for x in nodules]
    clean_paths = [os.path.join(image_directory, x.replace(".mha", ".png")) for x in cleans]
    for split_ix in range(splits):
        nodule_split = [x for x in nodule_paths if random.uniform(0, 1) <= 0.85]
        length_nodules = len(nodule_split) 
        clean_split = [x for x in clean_paths if random.uniform(0, 1) <= 0.85] # 1.2 X nodule_numbers
        length_clean = len(clean_split)
        should_be_clean_length = len(nodule_split) * ratio_clean_nodules
        for subsplit_ix in range(subsplits):
            clean_subsplit = random.sample(clean_split, should_be_clean_length)
            txt_filename = "train_V1_" + str(split_ix) + str(subsplit_ix) + ".txt"
            txt_filename = os.path.join(yaml_dir, txt_filename)
            text_train = "\n".join(nodule_split + clean_subsplit)
            with open(txt_filename, "w") as fhandle:
                fhandle.write(text_train)
            
            yaml_filename = os.path.join(yaml_dir, "nodule_V1_" + str(split_ix) + str(subsplit_ix) + ".yaml")
            data_yaml = dict(
                train = txt_filename ,
                val   = txt_filename,
                nc    = 1,
                names = ["nodule"]
                )
            with open(yaml_filename, 'w') as outfile:
                yaml.dump(data_yaml, outfile, default_flow_style=False)
    
def train_competition(yaml_dir):
    d = {}
    for ix, yaml_file in enumerate(os.listdir(yaml_dir)):
        if yaml_file.endswith(".yaml") == False:
            continue
        print("Training has started for yaml {}".format(ix))
        yaml_path =os.path.join(yaml_dir, yaml_file)
        command = "python3.6 train.py --img 1024 --batch 8 --epochs 30 --data " + str(yaml_path) + " --weights yolov5x.pt --cache"
        os.system(command)
        print(command)
        
def train_ensemble(yaml_dir):
    pass

def output_weights(trained_dir, output_dir):
    for ix, dir_experiment in enumerate(os.listdir(trained_dir)):
        weights_path = os.path.join(trained_dir, dir_experiment, "weights", "last.pt")
        new_path = os.path.join(output_dir, "experiment_" + str(ix) + ".pt")
        shutil.copy(weights_path, new_path)
        
import SimpleITK
import numpy as np

from pandas import DataFrame
import torch
from evalutils import DetectionAlgorithm
from evalutils.validators import (
    UniquePathIndicesValidator,
    UniqueImagesValidator,
)
import json
from typing import Dict
import os
import itertools
from pathlib import Path

# This parameter adapts the paths between local execution and execution in docker. You can use this flag to switch between these two modes.
# For building your docker, set this parameter to True. If False, it will run process.py locally for test purposes.
execute_in_docker = True

class Noduledetection(DetectionAlgorithm):
    def __init__(self, input_dir, output_dir, train=False, retrain=False, retest=False):
        super().__init__(
            validators=dict(
                input_image=(
                    UniqueImagesValidator(),
                    UniquePathIndicesValidator(),
                )
            ),
            input_path = Path(input_dir),
            output_file = Path(os.path.join(output_dir,'nodules.json'))
        )
        
        #------------------------------- LOAD the model here ---------------------------------
        self.input_path, self.output_path = input_dir, output_dir
        self.model_paths = ["/opt/algorithm/yolo1.pt", "/opt/algorithm/yolo2.pt", "/opt/algorithm/yolo3.pt", "/opt/algorithm/yolo4.pt", "/opt/algorithm/yolo5.pt", "/opt/algorithm/yolo6.pt", "/opt/algorithm/yolo7.pt", "/opt/algorithm/yolo8.pt", "/opt/algorithm/yolo9.pt", "/opt/algorithm/yolo1-1.pt", "/opt/algorithm/yolo2-1.pt", "/opt/algorithm/yolo3-1.pt", "/opt/algorithm/yolo4-1.pt", "/opt/algorithm/yolo5-1.pt", "/opt/algorithm/yolo6-1.pt", "/opt/algorithm/yolo7-1.pt", "/opt/algorithm/yolo8-1.pt", "/opt/algorithm/yolo9-1.pt", "/opt/algorithm/yolo1-2.pt", "/opt/algorithm/yolo2-2.pt", "/opt/algorithm/yolo3-2.pt", "/opt/algorithm/yolo4-2.pt", "/opt/algorithm/yolo5-2.pt", "/opt/algorithm/yolo6-2.pt", "/opt/algorithm/yolo7-2.pt", "/opt/algorithm/yolo8-2.pt", "/opt/algorithm/yolo9-2.pt", "/opt/algorithm/yolo1f1.pt", "/opt/algorithm/yolo1f2.pt", "/opt/algorithm/yolo2f1.pt", "/opt/algorithm/yolo2f2.pt", "/opt/algorithm/yolo3f1.pt", "/opt/algorithm/yolo3f2.pt",]
        self.images_dir = "/opt/algorithm/yolo_dataset/images"
        self.labels_yolo_dir = "/opt/algorithm/yolo_dataset/labels"
        self.yaml_dir = "/opt/algorithm/yamls"
        self.trained_models_dir = "/opt/algorithm/runs/train"
        
        if os.path.isdir(self.images_dir) == False:
            os.makedirs(self.images_dir)
            
        if os.path.isdir(self.labels_yolo_dir) == False:
            os.makedirs(self.labels_yolo_dir)
        
        # add path for yaml and txt files
        # add path for directory with images (png format) files
        # add path for directory with labels file
        
        self.model = load_yolo_predictor(data="/opt/algorithm/nodule01.yaml",
                           weights=self.model_paths)
        
    def save(self):
        with open(str(self._output_file), "w") as f:
            json.dump(self._case_results[0], f)
    
    def process_case(self, *, idx, case):
        '''
        Read the input, perform model prediction and return the results. 
        The returned value will be saved as nodules.json by evalutils.
        process_case method of evalutils
        (https://github.com/comic/evalutils/blob/fd791e0f1715d78b3766ac613371c447607e411d/evalutils/evalutils.py#L225) 
        is overwritten here, so that it directly returns the predictions without changing the format.
        
        '''
        # Load and test the image for this case
        input_image, input_image_file_path = self._load_input_image(case=case)
        
        # Detect and score candidates
        scored_candidates = self.predict(input_image=input_image)
        
        # Write resulting candidates to nodules.json for this case
        return scored_candidates
    
    def train(self, num_epochs=1):
        input_dir = self.input_path
        mha_dir = os.path.join(self.input_path, "images")
        images_dir = self.images_dir
        labels_yolo_dir = self.labels_yolo_dir
        print("Starting the conversion from mha to png")
        convert_nha_to_png(mha_dir, images_dir) # done
        print("Making the annotations dir")
        make_yolo_format_files(input_dir, labels_yolo_dir) # done
        #make_split_v1(input_dir)
        #make_split_v2(input_dir)
        #make_split_v3(input_dir)
        train_competition(self.yaml_dir)
        return 0
    
    def format_to_GC(self, np_predictions, spacing):
        return 0
    
    def merge_dict(self, results):
        merged_d = {}
        for k in results[0].keys():
            merged_d[k] = list(itertools.chain(*[d[k] for d in results]))
        return merged_d
        
    def predict(self, *, input_image):
        mha_path = input_image
        results = []
        if mha_path.GetDimension() == 2:
            image, spacing = convert_file([mha_path], None, auto_contrast=True)
            output_file_name = "/opt/algorithm/tmp.png"
            sitk.WriteImage(image, output_file_name)
            np_rgb_image = cv2.imread(output_file_name)
            image_index = 0
            output = run(np_rgb_image, model=self.model, augment=True, spacing=spacing, image_index=image_index)
            results.append(output, model=self.model, augment=True, spacing=spacing)
            os.remove(output_file_name)
        elif mha_path.GetDimension() == 3:
            num_components = mha_path.GetSize()[2]
            for j in range(num_components):
                image, spacing = convert_file([mha_path[:, :, j]], None, auto_contrast=True)
                output_file_name = "/opt/algorithm/tmp.png"
                sitk.WriteImage(image, output_file_name)
                np_rgb_image = cv2.imread(output_file_name)
                output = run(np_rgb_image, model=self.model, augment=True, spacing=spacing, image_index=j)
                results.append(output)
                os.remove(output_file_name)
        return dict(type="Multiple 2D bounding boxes", boxes=[el for sub_list in results for el in sub_list], version={ "major": 1, "minor": 0 })
    
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        prog='process.py',
        description=
            'Reads all images from an input directory and produces '
            'results in an output directory')

    parser.add_argument('input_dir', help = "input directory to process")
    parser.add_argument('output_dir', help = "output directory generate result files in")
    parser.add_argument('--train', action='store_true', help = "Algorithm on train mode.")
    parser.add_argument('--retrain', action='store_true', help = "Algorithm on retrain mode (loading previous weights).")
    parser.add_argument('--retest', action='store_true', help = "Algorithm on evaluate mode after retraining.")

    parsed_args = parser.parse_args()  
    if (parsed_args.train or parsed_args.retrain):# train mode: retrain or train
        Noduledetection(parsed_args.input_dir, parsed_args.output_dir, parsed_args.train, parsed_args.retrain, parsed_args.retest).train()
    else:# test mode (test or retest)
        Noduledetection(parsed_args.input_dir, parsed_args.output_dir, retest=parsed_args.retest).process()
