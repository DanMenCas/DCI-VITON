
from parsehuman import Preprocessing

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import matplotlib.pyplot as plt
import numpy as np
import os
import random
import json
import cv2
import argparse

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
from torchvision.transforms import ToPILImage
from torchvision import models
from torchvision import transforms
from PIL import Image


def make_grid(N, iW, iH, device):
    grid_x = torch.linspace(-1.0, 1.0, iW).view(1, 1, iW, 1).expand(N, iH, -1, -1).to(device)
    grid_y = torch.linspace(-1.0, 1.0, iH).view(1, iH, 1, 1).expand(N, -1, iW, -1).to(device)
    grid = torch.cat([grid_x, grid_y], 3)
    return grid

def save_tensor_as_image(tensor, save_path):
    """
    Saves a [C, H, W] or [1, C, H, W] tensor as an image file.
    """
    if tensor.dim() == 4:
        tensor = tensor.squeeze(0)  # Remove batch dimension

    to_pil = ToPILImage()
    image = to_pil(tensor.cpu().clamp(0, 1))  # Clamp values to [0,1] if needed
    image.save(save_path)

def flow_loss(flow_list):

    loss_tv = 0

    for flow in flow_list:
      y_tv = torch.abs(flow[:, 1:, :, :] - flow[:, :-1, :, :]).mean()
      x_tv = torch.abs(flow[:, :, 1:, :] - flow[:, :, :-1, :]).mean()
      loss_tv = loss_tv + y_tv + x_tv

    return loss_tv

class Vgg19(nn.Module):
    def __init__(self, requires_grad=False):
        super(Vgg19, self).__init__()
        vgg_pretrained_features = models.vgg19(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out

class VGGLoss(nn.Module):
    def __init__(self,layids = None):
        super(VGGLoss, self).__init__()
        self.vgg = Vgg19()
        self.criterion = nn.L1Loss()
        self.weights = [1.0/32, 1.0/16, 1.0/8, 1.0/4, 1.0]
        self.layids = layids

    def forward(self, x, y):
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        loss = 0
        if self.layids is None:
            self.layids = list(range(len(x_vgg)))
        for i in self.layids:
            loss += self.weights[i] * self.criterion(x_vgg[i], y_vgg[i].detach())
        return loss

class ResNetEncoderBlock(nn.Module):

  def __init__(self, input_channels, output_channels, use_dropout=False, use_bn=True, down=True, up=False):
      super(ResNetEncoderBlock, self).__init__()

      if down:
          self.scale = nn.Conv2d(input_channels, output_channels, kernel_size=3, stride=2, padding=1)
      elif up:
          self.scale = nn.Sequential(
              nn.Upsample(scale_factor=2, mode='bilinear'),
              nn.Conv2d(input_channels, output_channels, kernel_size=3, padding=1)
          )
      else:
          self.scale = nn.Conv2d(input_channels, output_channels, kernel_size=1)

      self.activation = nn.ReLU()

      if use_bn:
          self.batchnorm = nn.InstanceNorm2d(output_channels)
      self.use_bn = use_bn

      if use_dropout:
          self.dropout = nn.Dropout()
      self.use_dropout = use_dropout

      self.conv_1 = nn.Conv2d(output_channels, output_channels, kernel_size=3, padding=1)

      self.conv_2 = nn.Conv2d(output_channels, output_channels, kernel_size=3, padding=1)

  def forward(self, x):

      residual = self.scale(x)
      conv1_x = self.conv_1(residual)
      if self.use_bn:
          conv1_x = self.batchnorm(conv1_x)
      if self.use_dropout:
          conv1_x = self.dropout(conv1_x)
      conv1_x = self.activation(conv1_x)
      conv2_x = self.conv_2(conv1_x)
      if self.use_bn:
          conv2_x = self.batchnorm(conv2_x)
      if self.use_dropout:
          conv2_x = self.dropout(conv2_x)

      return self.activation(conv2_x + residual)

class ClothingEncoder(nn.Module):

  def __init__(self, input_channels, output_channels):
      super(ClothingEncoder, self).__init__()
      self.resnet1 = ResNetEncoderBlock(input_channels, output_channels)
      self.resnet2 = ResNetEncoderBlock(output_channels, output_channels * 2)
      self.resnet3 = ResNetEncoderBlock(output_channels * 2, output_channels * 4)
      self.resnet4 = ResNetEncoderBlock(output_channels * 4, output_channels * 4)
      self.resnet5 = ResNetEncoderBlock(output_channels * 4, output_channels * 4)

  def forward(self, x):

      x0 = self.resnet1(x)
      x1 = self.resnet2(x0)
      x2 = self.resnet3(x1)
      x3 = self.resnet4(x2)
      x4 = self.resnet5(x3)

      return x0, x1, x2, x3, x4

class SegmentEncoder(nn.Module):

  def __init__(self, input_channels, output_channels):
      super(SegmentEncoder, self).__init__()
      self.resnet1 = ResNetEncoderBlock(input_channels, output_channels)
      self.resnet2 = ResNetEncoderBlock(output_channels, output_channels * 2)
      self.resnet3 = ResNetEncoderBlock(output_channels * 2, output_channels * 4)
      self.resnet4 = ResNetEncoderBlock(output_channels * 4, output_channels * 4)
      self.resnet5 = ResNetEncoderBlock(output_channels * 4, output_channels * 4)
      self.resnet6 = ResNetEncoderBlock(output_channels * 4, output_channels * 8, down=False)
      self.resnet7 = ResNetEncoderBlock(output_channels * 8, output_channels * 4, down=False, up=True)

  def forward(self, x):

      x0 = self.resnet1(x)
      x1 = self.resnet2(x0)
      x2 = self.resnet3(x1)
      x3 = self.resnet4(x2)
      x4 = self.resnet5(x3)
      x5 = self.resnet6(x4)
      x6 = self.resnet7(x5)

      return x0, x1, x2, x3, x4, x5, x6

class WarpingProcess(nn.Module):

  def __init__(self, oc):
      super(WarpingProcess, self).__init__()

      self.conv_after_concat_1 = nn.Conv2d(oc*8, 2, kernel_size=3, padding=1)

      self.convs_clothing = nn.ModuleList([
          nn.Conv2d(oc*4, oc*4, kernel_size=1),
          nn.Conv2d(oc*4, oc*4, kernel_size=1),
          nn.Conv2d(oc*2, oc*4, kernel_size=1),
          nn.Conv2d(oc, oc*4, kernel_size=1)
      ])

      self.convs_segment = nn.ModuleList([
          nn.Conv2d(oc*4, oc*4, kernel_size=3, padding=1),
          nn.Conv2d(oc*4, oc*4, kernel_size=3, padding=1),
          nn.Conv2d(oc*4, oc*4, kernel_size=3, padding=1),
          nn.Conv2d(oc*4, oc*4, kernel_size=3, padding=1)
      ])

      self.conv_after_concat_2 = nn.ModuleList([
          nn.Conv2d(oc*8, 2, kernel_size=3, padding=1),
          nn.Conv2d(oc*8, 2, kernel_size=3, padding=1),
          nn.Conv2d(oc*8, 2, kernel_size=3, padding=1),
          nn.Conv2d(oc*8, 2, kernel_size=3, padding=1)
      ])

      self.convs_segment_after_concat = nn.ModuleList([
          ResNetEncoderBlock(oc*12, oc*4, down=False, up=True),
          ResNetEncoderBlock(oc*12, oc*4, down=False, up=True),
          ResNetEncoderBlock(oc*10, oc*4, down=False, up=True),
          ResNetEncoderBlock(oc*9, oc*4, down=False, up=True)
      ])

      self.clothingencoder = ClothingEncoder(4, oc)
      self.posencoder = SegmentEncoder(6, oc)

  def forward(self, input_1, input_2, device):

    clothing_before_warp = []
    pose_after_concat = []
    clothingpose_before_warp = []

    flow_list = []

    ce_0, ce_1, ce_2, ce_3, ce_4 = self.clothingencoder(input_1)
    clothing_features = [ce_0, ce_1, ce_2, ce_3, ce_4]

    pe_0, pe_1, pe_2, pe_3, pe_4, pe_5, pe_6 = self.posencoder(input_2)
    pose_features = [pe_0, pe_1, pe_2, pe_3, pe_4, pe_5, pe_6]

    clothing_pose_last_feature = torch.cat([ce_4, pe_4], dim=1)

    conv_before_warp = self.conv_after_concat_1(clothing_pose_last_feature)

    for i in range(4):

      if i == 0:

        up = F.interpolate(clothing_features[4], scale_factor=2, mode='bilinear')

        conv_up = self.convs_clothing[i](clothing_features[3 - i]) + up

        clothing_before_warp.append(conv_up)

        up_flow = F.interpolate(conv_before_warp, scale_factor=2, mode='bilinear')

        grid = make_grid(1, up_flow.shape[2], up_flow.shape[3], device)

        flow_norm = torch.cat([up_flow[:, 0:1, :, :] / ((up_flow.shape[3] - 1.0) / 2.0), up_flow[:, 1:2, :, :] / ((up_flow.shape[2] - 1.0) / 2.0)], 1).permute(0, 2, 3, 1)
        warped_T1 = F.grid_sample(conv_up, grid + flow_norm, padding_mode='border')

        flow_list.append(flow_norm)

        pe_concat = torch.cat([pose_features[6], warped_T1, pose_features[3 - i]], dim=1)

        conv_pe_6_out = self.convs_segment[i](pose_features[6])

        con_pe_6_warp = torch.concat([conv_pe_6_out, warped_T1], dim=1)

        conv_pe6_warp_out = self.conv_after_concat_2[i](con_pe_6_warp)

        concat_up_conv_pe6_warp = up_flow + conv_pe6_warp_out

        clothingpose_before_warp.append(concat_up_conv_pe6_warp)

        pe_last_resblock = self.convs_segment_after_concat[i](pe_concat)

        pose_after_concat.append(pe_last_resblock)

      else:

        up = F.interpolate(clothing_before_warp[i - 1], scale_factor=2, mode='bilinear')

        conv_up = self.convs_clothing[i](clothing_features[3 - i]) + up

        clothing_before_warp.append(conv_up)

        up_flow = F.interpolate(clothingpose_before_warp[i - 1], scale_factor=2, mode='bilinear')

        grid = make_grid(1, up_flow.shape[2], up_flow.shape[3], device)

        flow_norm = torch.cat([up_flow[:, 0:1, :, :] / ((up_flow.shape[3] - 1.0) / 2.0), up_flow[:, 1:2, :, :] / ((up_flow.shape[2] - 1.0) / 2.0)], 1).permute(0, 2, 3, 1)
        warped_T1 = F.grid_sample(conv_up, grid + flow_norm, padding_mode='border')

        flow_list.append(flow_norm)

        pe_concat = torch.cat([pose_after_concat[i - 1], warped_T1, pose_features[3 - i]], dim=1)

        conv_pe_6_out = self.convs_segment[i](pose_after_concat[i - 1])

        con_pe_6_warp = torch.concat([conv_pe_6_out, warped_T1], dim=1)

        conv_pe6_warp_out = self.conv_after_concat_2[i](con_pe_6_warp)

        concat_up_conv_pe6_warp = up_flow + conv_pe6_warp_out

        clothingpose_before_warp.append(concat_up_conv_pe6_warp)

        pe_last_resblock = self.convs_segment_after_concat[i](pe_concat)

        pose_after_concat.append(pe_last_resblock)

    last_up = F.interpolate(clothingpose_before_warp[-1], scale_factor=2, mode='bilinear')

    grid = make_grid(1, last_up.shape[2], last_up.shape[3], device)

    flow_norm = torch.cat([last_up[:, 0:1, :, :] / ((last_up.shape[3] - 1.0) / 2.0), last_up[:, 1:2, :, :] / ((last_up.shape[2] - 1.0) / 2.0)], 1).permute(0, 2, 3, 1)
    warped_T1 = F.grid_sample(input_1, grid + flow_norm, padding_mode='border')

    flow_list.append(flow_norm)

    warped_c = warped_T1[:, :-1, :, :]
    warped_cm = warped_T1[:, -1:, :, :]

    return warped_c, warped_cm, flow_list

class WarpingCloth():

  def __init__(self, size, checkpoint_warping):

    self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    self.checkpoint_warping = checkpoint_warping

    self.size = size

    self.transform = transforms.Compose([
            transforms.Resize(self.size),        # Resize to fixed size
            transforms.ToTensor(),          # Converts to [C, H, W], values in [0, 1]
        ])

    self.generator = WarpingProcess(96).to(self.device)
    self.gen_opt = torch.optim.Adam(self.generator.parameters(), lr=0.00005, betas=(0.5, 0.999))

  def warped_cloth_into_agnostic(self, agnostic_image, warped_cloth_image, warped_cloth_mask):

    agnostic_image_warped_cloth = agnostic_image.clone()

    warped_cloth_mask[warped_cloth_mask > 0.5] = 1

    warped_cloth_mask = warped_cloth_mask.repeat(1, 3, 1, 1)

    agnostic_image_warped_cloth[warped_cloth_mask == 1] = warped_cloth_image[warped_cloth_mask == 1]

    return agnostic_image_warped_cloth

  def warping(self, checkpoint_warping):

    self.generator.load_state_dict(self.checkpoint_warping['model_state_dict'])
    self.gen_opt.load_state_dict(self.checkpoint_warping['optimizer_state_dict'])

    warped_c, warped_cm, flow_list = self.generator(self.input_1, self.input_2, self.device)

    new_im = self.warped_cloth_into_agnostic(self.agnostic, warped_c, warped_cm)

    im_reshaped = F.interpolate(new_im, size=(512, 512), mode='bilinear', align_corners=False)

    return im_reshaped

  def __call__(self, person_img, cloth_img):

    img_preprocess = Preprocessing(person_img, cloth_img)

    open_pose = img_preprocess.open_pose()[:, :, ::-1]

    key_points = img_preprocess.key_points()

    parse_human = img_preprocess.parse_human()

    agnostic, agnostic_mask = img_preprocess.get_agnostic_and_mask()

    parse_agnostic = img_preprocess.get_parse_agnostic()
    parse_agnostic[parse_agnostic != parse_human] = 0

    cloth_mask = img_preprocess.cloth_mask()

    self.cloth = self.transform(img_preprocess.cloth_img).unsqueeze(0).to(self.device)
    self.cloth_mask = self.transform(Image.fromarray(cloth_mask)).unsqueeze(0).to(self.device)
    self.dense_pose = self.transform(Image.fromarray(open_pose)).unsqueeze(0).to(self.device)
    self.parse_agnostic = self.transform(Image.fromarray(parse_agnostic)).unsqueeze(0).to(self.device)
    self.agnostic = self.transform(Image.fromarray(agnostic)).unsqueeze(0).to(self.device)
    # This image is just converted from array to image to be the input of DDIM
    self.agnostic_mask = Image.fromarray(agnostic_mask)

    self.input_1 = torch.cat([self.cloth, self.cloth_mask[:, 0:1, ...]], dim=1)
    # The parse agnostic image is multiplied by 15 for have the correct normalization.
    self.input_2 = torch.cat([self.dense_pose, (self.parse_agnostic)*15], dim=1)

    output = self.warping(self.checkpoint_warping)

    return output

