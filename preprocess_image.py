import sys
sys.path.append("./DensePose")
import argparse
import glob
import logging
import os
from typing import Any, ClassVar, Dict, List
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt

from detectron2.config import CfgNode, get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.engine.defaults import DefaultPredictor

from densepose import add_densepose_config
from densepose.structures import DensePoseChartPredictorOutput, DensePoseEmbeddingPredictorOutput
from densepose.utils.logger import verbosity_to_level
from densepose.vis.base import CompoundVisualizer
from densepose.vis.bounding_box import ScoredBoundingBoxVisualizer
from densepose.vis.densepose_outputs_vertex import (
    DensePoseOutputsTextureVisualizer,
    DensePoseOutputsVertexVisualizer,
    get_texture_atlases,
)
from densepose.vis.densepose_results import (
    DensePoseResultsContourVisualizer,
    DensePoseResultsFineSegmentationVisualizer,
    DensePoseResultsUVisualizer,
    DensePoseResultsVVisualizer,
)
from densepose.vis.densepose_results_textures import (
    DensePoseResultsVisualizerWithTexture,
    get_texture_atlas,
)
from densepose.vis.extractor import (
    CompoundExtractor,
    DensePoseOutputsExtractor,
    DensePoseResultExtractor,
    create_extractor,
)


from transformers import SegformerImageProcessor, AutoModelForSemanticSegmentation
import torch.nn as nn


class preprocess_image():
    def __init__(self):
        self.dp_cfg = self.get_densepose_config()
        self.densepose_predictor = DefaultPredictor(self.dp_cfg)
        self.seg_processor = SegformerImageProcessor.from_pretrained("mattmdjaga/segformer_b2_clothes")
        self.seg_model = AutoModelForSemanticSegmentation.from_pretrained("mattmdjaga/segformer_b2_clothes")
        
    
    def get_densepose_config(self):
        config_fpath = "./DensePose/configs/densepose_rcnn_R_50_FPN_s1x.yaml"
        model_fpath = "https://dl.fbaipublicfiles.com/densepose/densepose_rcnn_R_50_FPN_s1x/165712039/model_final_162be9.pkl"
        opts = []
        cfg = get_cfg()
        add_densepose_config(cfg)
        cfg.merge_from_file(config_fpath)
        cfg.merge_from_list(opts)
        if opts:
            cfg.merge_from_list(opts)
        cfg.MODEL.WEIGHTS = model_fpath
        cfg.freeze()
        return cfg
    
    def get_densepose_output(self, image, outputs):
        COMMAND: ClassVar[str] = "show"
        VISUALIZERS: ClassVar[Dict[str, object]] = {
            "dp_contour": DensePoseResultsContourVisualizer,
            "dp_segm": DensePoseResultsFineSegmentationVisualizer,
            "dp_u": DensePoseResultsUVisualizer,
            "dp_v": DensePoseResultsVVisualizer,
            "dp_iuv_texture": DensePoseResultsVisualizerWithTexture,
            "dp_cse_texture": DensePoseOutputsTextureVisualizer,
            "dp_vertex": DensePoseOutputsVertexVisualizer,
            "bbox": ScoredBoundingBoxVisualizer,
        }

        vis_specs = ['dp_contour', 'dp_u', 'dp_v', 'bbox']
        visualizers = []
        extractors = []
        for vis_spec in vis_specs:
            texture_atlas = get_texture_atlas(None)
            texture_atlases_dict = get_texture_atlases(None)
            vis = VISUALIZERS[vis_spec](
                cfg=self.dp_cfg,
                texture_atlas=texture_atlas,
                texture_atlases_dict=texture_atlases_dict,
            )
            visualizers.append(vis)
            extractor = create_extractor(vis)
            extractors.append(extractor)
        visualizer = CompoundVisualizer(visualizers)
        extractor = CompoundExtractor(extractors)
        context = {
            "extractor": extractor,
            "visualizer": visualizer,
        }

        visualizer = context["visualizer"]
        extractor = context["extractor"]
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = np.tile(image[:, :, np.newaxis], [1, 1, 3])
        data = extractor(outputs)
        image_vis = visualizer.visualize(image, data)    
        return image_vis
    
    def get_densepose_prediction(self, img):
        with torch.no_grad():
            output = self.densepose_predictor(img)["instances"]
        densepose = self.get_densepose_output(img, output)
        return densepose
        
        
    def get_seg_prediction(self, pil_image):
        if pil_image.mode != "RGB":
            image = pil_image.convert("RGB")
        else:
            image = pil_image
        inputs = self.seg_processor(images=image, return_tensors="pt")

        outputs = self.seg_model(**inputs)
        logits = outputs.logits.cpu()

        upsampled_logits = nn.functional.interpolate(
            logits,
            size=image.size[::-1],
            mode="bilinear",
            align_corners=False,
        )

        pred_seg = upsampled_logits.argmax(dim=1)[0]
        
        plt.imshow(pred_seg)
        plt.savefig('person_seg.png')
        plt.show()

        return pred_seg