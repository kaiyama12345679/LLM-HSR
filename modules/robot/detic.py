# import some common libraries
import sys
import numpy as np
import os, json, random
import copy
from typing import List
import torch
import hsrb_interface
import cv2
import message_filters
import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image

#The Detic Project Path
DETIC_PATH="/root/Detic/"
sys.path.append(DETIC_PATH)
sys.path.insert(0, os.path.join(DETIC_PATH, "third_party/CenterNet2/"))

import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog

from centernet.config import add_centernet_config
from detic.config import add_detic_config
from detic.modeling.utils import reset_cls_test
from detic.modeling.text.text_encoder import build_text_encoder

cfg = get_cfg()
add_centernet_config(cfg)
add_detic_config(cfg)
cfg.merge_from_file(DETIC_PATH + "configs/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.yaml")
cfg.MODEL.WEIGHTS = 'https://dl.fbaipublicfiles.com/detic/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.pth'
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
cfg.MODEL.ROI_BOX_HEAD.ZEROSHOT_WEIGHT_PATH = 'rand'
cfg.MODEL.ROI_HEADS.ONE_CLASS_PER_PROPOSAL = True # For better visualization purpose. Set to False for all classes.
# cfg.MODEL.DEVICE='cpu' # uncomment this to use cpu-only mode.

BUILDIN_CLASSIFIER = {
    'lvis': 'datasets/metadata/lvis_v1_clip_a+cname.npy',
    'objects365': 'datasets/metadata/o365_clip_a+cnamefix.npy',
    'openimages': 'datasets/metadata/oid_clip_a+cname.npy',
    'coco': 'datasets/metadata/coco_clip_a+cname.npy',
}

BUILDIN_METADATA_PATH = {
    'lvis': 'lvis_v1_val',
    'objects365': 'objects365_v2_val',
    'openimages': 'oid_val_expanded',
    'coco': 'coco_2017_val',
}
def get_clip_embeddings(vocabulary, prompt='a '):
    text_encoder = build_text_encoder(pretrain=True)
    text_encoder.eval()
    texts = [prompt + x for x in vocabulary]
    emb = text_encoder(texts).detach().permute(1, 0).contiguous().cpu()
    return emb

class DeticPredictor:
    def __init__(self, vocabulary: str | List[str] = 'lvis'):
        #  vocabulary = 'lvis' # change to 'lvis', 'objects365', 'openimages', 'coco' or 'List of your own classes'
        self.predictor = DefaultPredictor(cfg)
        if vocabulary not in BUILDIN_CLASSIFIER:
            self.vocabulary = vocabulary
            self.metadata = MetadataCatalog.get("__unused")
            self.metadata.thing_classes = vocabulary
            self.classifier = get_clip_embeddings(self.metadata.thing_classes)
            self.num_classes = len(self.metadata.thing_classes)
            reset_cls_test(self.predictor.model, self.classifier, self.num_classes)
        else:
            self.vocabulary = vocabulary
            self.metadata = MetadataCatalog.get(BUILDIN_METADATA_PATH[vocabulary])
            self.classifier = DETIC_PATH + BUILDIN_CLASSIFIER[vocabulary]
            self.num_classes = len(self.metadata.thing_classes)
            reset_cls_test(self.predictor.model, self.classifier, self.num_classes)

        head_rgb_sub = message_filters.Subscriber('/hsrb/head_rgbd_sensor/rgb/image_raw', Image)
        head_depth_sub = message_filters.Subscriber('/hsrb/head_rgbd_sensor/depth_registered/image', Image)
        hand_rgb_sub = message_filters.Subscriber('/hsrb/hand_camera/image_raw', Image)
        self.detection_result_pub_head = rospy.Publisher('/detic_image_node/detection_result_head', Image, queue_size=10)
        self.detection_result_pub_hand = rospy.Publisher('/detic_image_node/detection_result_hand', Image, queue_size=10)

        message_filters.ApproximateTimeSynchronizer([head_rgb_sub, head_depth_sub, hand_rgb_sub], 10, 1.0).registerCallback(self._callback)

        self.bridge = CvBridge()
        self.head_rgb_image, self.head_depth_image, self.hand_rgb_image = None, None, None

        self.head_objects = []
        self.hand_objects = []

        self.should_stop = False


    def _callback(self, head_rgb_data, head_depth_data, hand_rgb_data):
        head_rgb_array = self.bridge.imgmsg_to_cv2(head_rgb_data, 'bgr8')
        head_rgb_array = cv2.cvtColor(head_rgb_array, cv2.COLOR_BGR2RGB)
        self.head_rgb_image = head_rgb_array

        head_depth_array = self.bridge.imgmsg_to_cv2(head_depth_data, 'passthrough')
        self.head_depth_image = head_depth_array

        hand_rgb_array = self.bridge.imgmsg_to_cv2(hand_rgb_data, 'bgr8')
        hand_rgb_array = cv2.cvtColor(hand_rgb_array, cv2.COLOR_BGR2RGB)
        self.hand_rgb_image = hand_rgb_array

    def _publish_result(self, image, outputs, pub):
        v = Visualizer(image[:, :, ::-1], self.metadata)
        out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        output = out.get_image()[:, :, ::-1]
        output = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
        pub.publish(self.bridge.cv2_to_imgmsg(output, 'bgr8'))

    def process(self):
        while not rospy.is_shutdown() or not self.should_stop:
            if self.head_rgb_image is None or self.head_depth_image is None or self.hand_rgb_image is None:
                continue
            tmp_head = copy.deepcopy(self.head_rgb_image)
            tmp_hand = copy.deepcopy(self.hand_rgb_image)
            # inference
            with torch.no_grad():
                outputs_from_head = self.predictor(self.head_rgb_image)
                outputs_from_hand = self.predictor(self.hand_rgb_image)
            
            self._publish_result(tmp_head, outputs_from_head, self.detection_result_pub_head)
            self._publish_result(tmp_hand, outputs_from_hand, self.detection_result_pub_hand)

            hand_rgb_size = self.hand_rgb_image.shape
            head_rgb_size = self.head_rgb_image.shape

            self.head_objects = []
            self.hand_objects = []

            # plot bounding box and control movement
            for idx in range(len(outputs_from_head["instances"].pred_classes)):
                object_cls = self.metadata.thing_classes[outputs_from_head["instances"].pred_classes[idx]]

                if  not (object_cls == "book" or object_cls == "person"):
                    continue

                box = outputs_from_head["instances"].pred_boxes[idx].tensor.cpu().numpy()[0]
                min_x, min_y, max_x, max_y = int(box[0]), int(box[1]), int(box[2]), int(box[3])
                info = {"cls": object_cls, "location": (min_x, min_y, max_x, max_y), "depth": self.head_depth_image[int(min_y):int(max_y), int(min_x):int(max_x)]}
                self.head_objects.append(info)
        
            for idx in range(len(outputs_from_hand["instances"].pred_classes)):
                object_cls = self.metadata.thing_classes[outputs_from_head["instances"].pred_classes[idx]]

                if  not (object_cls == "book" or object_cls == "person"):
                    continue

                box = outputs_from_head["instances"].pred_boxes[idx].tensor.cpu().numpy()[0]
                min_x, min_y, max_x, max_y = int(box[0]), int(box[1]), int(box[2]), int(box[3])
                info = {"cls": object_cls, "location": (min_x, min_y, max_x, max_y)}
                self.hand_objects.append(info)


if __name__ == '__main__':
    dd = DeticPredictor()
    try:
        dd.process()
    except rospy.ROSInitException:
        pass