
import os
os.environ["NUMBA_DISABLE_JIT"] = "1"

import matplotlib.pyplot as plt
import json
import numpy as np
import torch, cv2
import argparse
import torch.nn.functional as F

from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.utils.visualizer import Visualizer
from rembg import remove, new_session
from tqdm import tqdm
from PIL import Image, ImageDraw
from os import path as osp
from transformers import SegformerImageProcessor, AutoModelForSemanticSegmentation
from detectron2.data.detection_utils import read_image
from densepose.config import add_densepose_config
from densepose.vis.densepose_results import DensePoseResultsFineSegmentationVisualizer
from densepose.vis.bounding_box import ScoredBoundingBoxVisualizer
from densepose.vis.base import CompoundVisualizer
from densepose.vis.extractor import create_extractor, CompoundExtractor

class Preprocessing():

  def __init__(self, person_image, cloth_image):

    self.img = person_image
    self.cloth_img = cloth_image
    self.device = "cuda" if torch.cuda.is_available() else "cpu"


  def key_points(self):

    self.image = np.array(self.img)

    # Load pretrained COCO keypoint model
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml")
    cfg.MODEL.DEVICE = self.device
    predictor = DefaultPredictor(cfg)

    outputs = predictor(self.image)
    keypoints = outputs["instances"].pred_keypoints.cpu().numpy()

    #0.Nose
    #1.Right eye
    #2.Left eye
    #3.Right ear
    #4.Left ear
    #5.Right shoulder
    #6.Left Shoulder
    #7.Right elbow
    #8.Left elbow
    #9.Right wrist
    #10.Left wrist
    #11.Right hip
    #12.Left hip
    #13.Right knee
    #14.Left Knee
    #15.Right Wrist
    #16.Left Wrist

    kpoints = keypoints[0][:, 0:2]

    keypoints_chest = (kpoints[5]+kpoints[6])/2

    keypoints_chest_reshaped = keypoints_chest.reshape(1, 2)

    kpoints_chest = np.vstack([kpoints, keypoints_chest_reshaped])

    # Visualize
    # v = Visualizer(image[:, :, ::-1], scale=1.0)
    # out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    # cv2_imshow(out.get_image()[:, :, ::-1])

    return kpoints_chest

  def parse_human(self):

    # ---- Config ----
    model_name = "yolo12138/segformer-b2-human-parse-24"

    # ---- Load model ----
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = SegformerImageProcessor.from_pretrained(model_name)
    model = AutoModelForSemanticSegmentation.from_pretrained(model_name).to(device).eval()

    # ---- Load and preprocess image ----
    inputs = processor(images=self.img, return_tensors="pt").to(device)

    # ---- Inference ----
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

    # ---- Resize to original size ----
    orig_size = self.img.size  # (width, height)
    logits_upsampled = F.interpolate(
        logits, size=(orig_size[1], orig_size[0]),
        mode="bilinear", align_corners=False
    )
    seg_map = logits_upsampled.argmax(dim=1)[0].cpu().numpy()

    # ---- LIP 20-class color palette ---- (85,255,170), (170,255,85)
    # (face=blue, hair=red, upper-clothes=orange, arms=cyan, pants=teal, etc.)
    LIP_COLORS = np.array([
        [0, 0, 0],       # 0 background
        [1, 1, 1],       # 1 hat
        [2, 2, 2],       # 2 hair
        [3, 3, 3],       # 3 glove
        [4, 4, 4],       # 4 glasses
        [5, 5, 5],       # 5 upper clothes
        [6, 6, 6],       # 6 dress
        [7, 7, 7],       # 7 coat
        [8, 8, 8],       # 8 socks
        [9, 9, 9],       # 9 left_pants
        [9, 9, 9],       # 10 right_pants
        [10, 10, 10],    # 11 skin_around_neck_region
        [11, 11, 11],    # 12 scarf
        [12, 12, 12],    # 13 skirt
        [13, 13, 13],    # 14 Face
        [14, 14, 14],    # 15 left arm
        [15, 15, 15],    # 16 right arm
        [16, 16, 16],    # 17 left leg
        [17, 17, 17],    # 18 right leg
        [18, 18, 18],    # 19 left shoe
        [19, 19, 19],    # 20 right shoe
        [5, 5, 5],       # 21 left_sleeve_for_upper
        [5, 5, 5],       # 22 right_sleeve_for_upper
        [0, 0, 0]        # 23 bag
    ], dtype=np.uint8)

    # ---- Apply palette ----
    seg_colored = LIP_COLORS[seg_map % len(LIP_COLORS)]

    # # ---- Display ----
    # fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    # axes[0].imshow(img_np)
    # axes[0].set_title("Original")
    # axes[0].axis("off")

    # axes[1].imshow(seg_colored)
    # axes[1].set_title("LIP-style Segmentation Map")
    # axes[1].axis("off")

    # axes[2].imshow(overlay)
    # axes[2].set_title("Overlay")
    # axes[2].axis("off")

    # plt.tight_layout()
    # plt.show()

    return seg_colored

  def get_agnostic_and_mask(self):

    im = self.img
    im_parse = self.parse_human()[:, :, 0]
    pose_data = self.key_points()

    parse_array = np.array(im_parse)
    parse_head = ((parse_array == 4).astype(np.float32) +
                  (parse_array == 13).astype(np.float32))
    parse_lower = ((parse_array == 9).astype(np.float32) +
                  (parse_array == 12).astype(np.float32) +
                  (parse_array == 16).astype(np.float32) +
                  (parse_array == 17).astype(np.float32) +
                  (parse_array == 18).astype(np.float32) +
                  (parse_array == 19).astype(np.float32))

    # Initialize agnostic image and mask
    agnostic = im.copy()
    mask = Image.new('L', im.size, 0)  # Binary mask: black background
    agnostic_draw = ImageDraw.Draw(agnostic)
    mask_draw = ImageDraw.Draw(mask)

    # Normalize arm lengths
    length_a = np.linalg.norm(pose_data[5] - pose_data[6])
    length_b = np.linalg.norm(pose_data[11] - pose_data[12])
    point = (pose_data[12] + pose_data[11]) / 2
    pose_data[12] = point + (pose_data[12] - point) / length_b * length_a
    pose_data[11] = point + (pose_data[11] - point) / length_b * length_a

    r = int(length_a / 16) + 1

    # Mask torso
    for i in [12, 11]:
        pointx, pointy = pose_data[i]
        agnostic_draw.ellipse((pointx-r*3, pointy-r*6, pointx+r*3, pointy+r*6), 'gray', 'gray')
        mask_draw.ellipse((pointx-r*3, pointy-r*6, pointx+r*3, pointy+r*6), 255, 255)
    agnostic_draw.line([tuple(pose_data[i]) for i in [6, 12]], 'gray', width=r*6)
    mask_draw.line([tuple(pose_data[i]) for i in [6, 12]], 255, width=r*6)
    agnostic_draw.line([tuple(pose_data[i]) for i in [5, 11]], 'gray', width=r*6)
    mask_draw.line([tuple(pose_data[i]) for i in [5, 11]], 255, width=r*6)
    agnostic_draw.line([tuple(pose_data[i]) for i in [12, 11]], 'gray', width=r*12)
    mask_draw.line([tuple(pose_data[i]) for i in [12, 11]], 255, width=r*12)
    agnostic_draw.polygon([tuple(pose_data[i]) for i in [6, 5, 11, 12]], 'gray', 'gray')
    mask_draw.polygon([tuple(pose_data[i]) for i in [6, 5, 11, 12]], 255, 255)

    # Mask neck
    pointx, pointy = pose_data[17]
    agnostic_draw.rectangle((pointx-r*5, pointy-r*9, pointx+r*5, pointy), 'gray', 'gray')
    mask_draw.rectangle((pointx-r*5, pointy-r*9, pointx+r*5, pointy), 255, 255)

    # Mask arms
    agnostic_draw.line([tuple(pose_data[i]) for i in [6, 5]], 'gray', width=r*12)
    mask_draw.line([tuple(pose_data[i]) for i in [6, 5]], 255, width=r*12)
    for i in [6, 5]:
        pointx, pointy = pose_data[i]
        agnostic_draw.ellipse((pointx-r*5, pointy-r*6, pointx+r*5, pointy+r*6), 'gray', 'gray')
        mask_draw.ellipse((pointx-r*5, pointy-r*6, pointx+r*5, pointy+r*6), 255, 255)
    for i in [8, 10, 7, 9]:
        if (pose_data[i-1, 0] == 0.0 and pose_data[i-1, 1] == 0.0) or (pose_data[i, 0] == 0.0 and pose_data[i, 1] == 0.0):
            continue
        agnostic_draw.line([tuple(pose_data[j]) for j in [i - 2, i]], 'gray', width=r*10)
        mask_draw.line([tuple(pose_data[j]) for j in [i - 2, i]], 255, width=r*10)
        pointx, pointy = pose_data[i]
        agnostic_draw.ellipse((pointx-r*5, pointy-r*5, pointx+r*5, pointy+r*5), 'gray', 'gray')
        mask_draw.ellipse((pointx-r*5, pointy-r*5, pointx+r*5, pointy+r*5), 255, 255)

    # Handle arm masks for parsing
    for parse_id, pose_ids in [(14, [5, 7, 9]), (15, [6, 8, 10])]:
        mask_arm = Image.new('L', (768, 1024), 255)  # White background
        mask_arm_draw = ImageDraw.Draw(mask_arm)
        pointx, pointy = pose_data[pose_ids[0]]
        mask_arm_draw.ellipse((pointx-r*5, pointy-r*6, pointx+r*5, pointy+r*6), 0, 0)
        for i in pose_ids[1:]:
            if (pose_data[i-1, 0] == 0.0 and pose_data[i-1, 1] == 0.0) or (pose_data[i, 0] == 0.0 and pose_data[i, 1] == 0.0):
                continue
            mask_arm_draw.line([tuple(pose_data[j]) for j in [i - 2, i]], 0, width=r*10)
            pointx, pointy = pose_data[i]
            if i != pose_ids[-1]:
                mask_arm_draw.ellipse((pointx-r*5, pointy-r*5, pointx+r*5, pointy+r*5), 0, 0)
        mask_arm_draw.ellipse((pointx-r*4, pointy-r*4, pointx+r*4, pointy+r*4), 0, 0)

        parse_arm = (np.array(mask_arm) / 255) * (parse_array == parse_id).astype(np.float32)
        agnostic.paste(im, None, Image.fromarray(np.uint8(parse_arm * 255), 'L'))

    # Paste head and lower body
    agnostic.paste(im, None, Image.fromarray(np.uint8(parse_head * 255), 'L'))
    agnostic.paste(im, None, Image.fromarray(np.uint8(parse_lower * 255), 'L'))

    # Convert mask and agnostic to NumPy arrays
    parse_array[~np.isin(parse_array, [4, 13, 9, 12, 16, 17, 18, 19])] = 1
    parse_array[np.isin(parse_array, [4, 13, 9, 12, 16, 17, 18, 19])] = 0

    mask = np.array(mask)  # Binary mask (0 for background, 255 for masked areas)
    agnostic = np.array(agnostic)

    mask = mask * parse_array

    return agnostic, mask

  def get_parse_agnostic(self):

    im_parse = Image.fromarray(self.parse_human()[:, :, 0])
    pose_data = self.key_points()

    parse_array = np.array(im_parse)
    parse_head = ((parse_array == 2).astype(np.float32) +
                  (parse_array == 4).astype(np.float32)+
                  (parse_array == 13).astype(np.float32))
    parse_lower = ((parse_array == 9).astype(np.float32) +
                  (parse_array == 12).astype(np.float32) +
                  (parse_array == 16).astype(np.float32) +
                  (parse_array == 17).astype(np.float32) +
                  (parse_array == 18).astype(np.float32) +
                  (parse_array == 19).astype(np.float32))

    agnostic = im_parse.copy()
    agnostic_draw = ImageDraw.Draw(agnostic)

    length_a = np.linalg.norm(pose_data[5] - pose_data[6])
    length_b = np.linalg.norm(pose_data[11] - pose_data[12])
    point = (pose_data[12] + pose_data[11]) / 2
    pose_data[12] = point + (pose_data[12] - point) / length_b * length_a
    pose_data[11] = point + (pose_data[11] - point) / length_b * length_a

    r = int(length_a / 16) + 1

    # mask torso
    for i in [12, 11]:
        pointx, pointy = pose_data[i]
        agnostic_draw.ellipse((pointx-r*3, pointy-r*6, pointx+r*3, pointy+r*6), 'gray', 'gray')
    agnostic_draw.line([tuple(pose_data[i]) for i in [6, 12]], 'gray', width=r*6)
    agnostic_draw.line([tuple(pose_data[i]) for i in [5, 11]], 'gray', width=r*6)
    agnostic_draw.line([tuple(pose_data[i]) for i in [12, 11]], 'gray', width=r*12)
    agnostic_draw.polygon([tuple(pose_data[i]) for i in [6, 5, 11, 12]], 'gray', 'gray')

    #mask neck
    pointx, pointy = pose_data[17]
    agnostic_draw.rectangle((pointx-r*5, pointy-r*9, pointx+r*5, pointy), 'gray', 'gray')

    # mask arms
    agnostic_draw.line([tuple(pose_data[i]) for i in [6, 5]], 'gray', width=r*12)
    for i in [6, 5]:
        pointx, pointy = pose_data[i]
        agnostic_draw.ellipse((pointx-r*5, pointy-r*6, pointx+r*5, pointy+r*6), 'gray', 'gray')
    for i in [8, 10, 7, 9]:
        if (pose_data[i-1, 0] == 0.0 and pose_data[i-1, 1] == 0.0) or (pose_data[i, 0] == 0.0 and pose_data[i, 1] == 0.0):
            continue
        agnostic_draw.line([tuple(pose_data[j]) for j in [i - 2, i]], 'gray', width=r*10)
        pointx, pointy = pose_data[i]
        agnostic_draw.ellipse((pointx-r*5, pointy-r*5, pointx+r*5, pointy+r*5), 'gray', 'gray')

    for parse_id, pose_ids in [(14, [5, 7, 9]), (15, [6, 8, 10])]:
        mask_arm = Image.new('L', (768, 1024), 'white')
        mask_arm_draw = ImageDraw.Draw(mask_arm)
        pointx, pointy = pose_data[pose_ids[0]]
        mask_arm_draw.ellipse((pointx-r*5, pointy-r*6, pointx+r*5, pointy+r*6), 'black', 'black')
        for i in pose_ids[1:]:
            if (pose_data[i-1, 0] == 0.0 and pose_data[i-1, 1] == 0.0) or (pose_data[i, 0] == 0.0 and pose_data[i, 1] == 0.0):
                continue
            mask_arm_draw.line([tuple(pose_data[j]) for j in [i - 2, i]], 'black', width=r*10)
            pointx, pointy = pose_data[i]
            if i != pose_ids[-1]:
                mask_arm_draw.ellipse((pointx-r*5, pointy-r*5, pointx+r*5, pointy+r*5), 'black', 'black')
        mask_arm_draw.ellipse((pointx-r*4, pointy-r*4, pointx+r*4, pointy+r*4), 'black', 'black')

        parse_arm = (np.array(mask_arm) / 255) * (parse_array == parse_id).astype(np.float32)
        agnostic.paste(im_parse, None, Image.fromarray(np.uint8(parse_arm * 255), 'L'))

    agnostic.paste(im_parse, None, Image.fromarray(np.uint8(parse_head * 255), 'L'))
    agnostic.paste(im_parse, None, Image.fromarray(np.uint8(parse_lower * 255), 'L'))

    parse_agnostic = np.array(agnostic)

    #This is for create the same img with three channels

    parse_agnostic_3_ch = np.repeat(parse_agnostic[:, :, np.newaxis], repeats=3, axis=2)

    return parse_agnostic_3_ch

  def open_pose(self):

    img = np.array(self.img)[:, :, ::-1]

    cfg = get_cfg()
    add_densepose_config(cfg)
    # add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
    cfg.merge_from_file("densepose/configs/densepose_rcnn_R_50_FPN_s1x.yaml")
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.8  # set threshold for this model
    # Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
    cfg.MODEL.WEIGHTS = "https://dl.fbaipublicfiles.com/densepose/densepose_rcnn_R_50_FPN_s1x/165712039/model_final_162be9.pkl"
    #cfg.MODEL.WEIGHTS = "https://dl.fbaipublicfiles.com/densepose/cse/densepose_rcnn_R_50_FPN_s1x/251155172/model_final_c4ea5f.pkl"
    # Set the device to CPU if a CUDA enabled GPU is not available
    cfg.MODEL.DEVICE = self.device
    predictor = DefaultPredictor(cfg)
    outputs = predictor(img)['instances']

    dp_segm_vis = DensePoseResultsFineSegmentationVisualizer(cfg=cfg)
    visualizer = CompoundVisualizer([dp_segm_vis])
    extractor = CompoundExtractor([
        create_extractor(dp_segm_vis),
    ])

    image_zero = np.zeros_like(img)
    data = extractor(outputs)
    image_vis = visualizer.visualize(image_zero, data)

    # Normalize the image to the range of 0 to 255
    # First, ensure the image is a floating-point type for calculations
    image = image_vis.astype(np.float32)

    # Find the minimum and maximum pixel values
    min_val = np.min(image)
    max_val = np.max(image)

    # Normalize the image to the range [0, 1]
    normalized_image = (image - min_val) / (max_val - min_val)

    # Scale the normalized image to the range [0, 255] and convert to uint8
    image_normalized_255 = (normalized_image * 255).astype(np.uint8)

    # You can now use image_normalized_255 for display or further processing
    # For example, to display the normalized image:
    # cv2_imshow(image_normalized_255)

    return image_normalized_255

  def cloth_mask(self):

    input_img = self.cloth_img

    # Specify the model to use with GPU
    model_name = "u2net"  # u2net supports GPU acceleration
    session = new_session(model_name, providers=["CUDAExecutionProvider"])  # Enable GPU

    # Generate the mask (white for the object/cloth, black for background) using GPU
    mask = remove(input_img, session=session, only_mask=True)

    return np.array(mask)

