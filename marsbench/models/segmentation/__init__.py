"""
Segmentation models for Mars surface image segmentation tasks.
"""
from .BaseSegmentationModel import BaseSegmentationModel
from .DeepLab import DeepLab
from .DINOv2Segmentation import DINOv2Segmentation
from .DPT import DPT
from .Mask2Former import Mask2Former
from .Segformer import Segformer
from .UNet import UNet
#from .ViTMAE import ViTMAESemSeg
from .transformersDPT import TransformersDPT

__all__ = ["BaseSegmentationModel", "UNet", "DeepLab", "Segformer", "Mask2Former",
           "DPT", "TransformersDPT", "DINOv2Segmentation"]
