#Need to convert this cell to .py file
import os
import glob
import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torchvision.utils import save_image

from model.model import HumanSegment, HumanMatting
import utils
import inference


def Matting(image_dir):

  model = HumanMatting(backbone='resnet50')
  model = nn.DataParallel(model).cuda().eval()
  model.load_state_dict(torch.load("./pretrained/SGHM-ResNet50.pth"))

  image_list = sorted([*glob.glob(os.path.join(image_dir, '**', '*.jpg'), recursive=True),
                      *glob.glob(os.path.join(image_dir, '**', '*.png'), recursive=True)])

  num_image = len(image_list)

  for i in range(num_image):
      image_path = image_list[i]
      image_name = image_path[image_path.rfind('/')+1:image_path.rfind('.')]

      with Image.open(image_path) as img:
          img = img.convert("RGB")

      pred_alpha, pred_mask = inference.single_inference(model, img)

      # output_dir = result_dir + image_path[len(image_dir):image_path.rfind('/')]
      if not os.path.exists(output_matting):
          os.makedirs(output_matting)
      save_path = 'output_matting/fg_mask.png'
      Image.fromarray(((pred_alpha * 255).astype('uint8')), mode='L').save(save_path)
