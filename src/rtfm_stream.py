#!/usr/bin/env python3
"""OpenCV feature detectors with ros CompressedImage Topics in python.

This example subscribes to a ros topic containing sensor_msgs 
CompressedImage. It converts the CompressedImage into a numpy.ndarray, 
then detects and marks features in that image. It finally displays 
and publishes the new image - again as CompressedImage topic.
"""
__author__ = "Simon Haller <simon.haller at uibk.ac.at>"
__version__ = "0.1"
__license__ = "BSD"
# Python libs
import sys, time

# numpy, opencv
import numpy as np
import cv2

# Ros libraries
import roslib
import rospy

# Ros Messages
from sensor_msgs.msg import Image
from accident_prediction.msg import ImageStream
from std_msgs.msg import Float32MultiArray

# We do not use cv_bridge it does not support CompressedImage in python
from cv_bridge import CvBridge, CvBridgeError

# pytorch
import torch
from torch.utils.data import Dataset, DataLoader, IterableDataset
import torchvision.transforms as T

# other packages
from models.resnet import I3Res50
from models.rtfm import Model, Model2

VERBOSE = False

device = torch.device("cuda:0")

mean = [114.75, 114.75, 114.75]
std = [57.375, 57.375, 57.375]
it = 0


def i3_res50(num_classes):
    net = I3Res50(num_classes=num_classes, use_nl=False)
    state_dict = torch.load("ckpt/i3d_r50_kinetics.pth")
    net.load_state_dict(state_dict)
    return net


def process_feat(feat, length):
    new_feat = torch.zeros(
        (length, feat.shape[1]), dtype=torch.float32
    )  # UCF(32, 2048)
    r = torch.linspace(0, len(feat), length + 1, dtype=torch.int32)  # (33,)
    for i in range(length):
        if r[i] != r[i + 1]:
            new_feat[i, :] = torch.mean(feat[r[i] : r[i + 1], :], dim=0)
        else:
            new_feat[i, :] = feat[r[i], :]
    return new_feat


class AccidentPrediction:
    def __init__(self):
        self.bridge = CvBridge()
        self.publisher = rospy.Publisher("logits", Float32MultiArray)
        self.subscriber = rospy.Subscriber(
            "image_stream",
            ImageStream,
            self.callback,
            queue_size=10,
        )
        self.transform = T.Compose(
            [
                T.ToTensor(),
                T.Resize(256),
                T.TenCrop(224),
                T.Lambda(lambda crops: torch.stack([crop * 255 for crop in crops])),
                T.Normalize(mean, std),
            ]
        )
        if VERBOSE:
            print("subscribed to /image_stream")
            print(" to /image_stream")

    def callback(self, ros_data):
        global it
        it += 1
        if VERBOSE:
            print('received stream of type: "%s"' % ros_data.format)
        stream = ros_data.data
        print("len 50")
        net = i3_res50(400).to(device)
        model = Model(2048, 10)
        # ckpt = torch.load("ckpt/rtfmfinal.pkl")
        ckpt = torch.load("ckpt/rtfm1430-i3d.pkl")  # carla
        # ckpt = torch.load("ckpt/rtfm1535-i3d.pkl")  # ccd
        model.load_state_dict(ckpt)
        model = model.to(device)
        with torch.no_grad():
            net.eval()
            model.eval()
            video_data = []
            for i, img in enumerate(stream):
                img_np = self.bridge.imgmsg_to_cv2(img, "bgr8")
                cv2.imwrite("log/iter_{}_{}.jpg".format(it, i), img_np)
                video_data.append(self.transform(img_np))
            frames_list = torch.stack([clip for clip in video_data])
            frames_list = frames_list.unsqueeze(0).to(device)
            snippets_list = frames_list.unfold(1, 10, 10).permute(0, 1, 2, 3, 6, 4, 5)
            inp = {"frames": snippets_list}
            features, _ = net(inp)

            # data
            features = features[0]
            divided_features = []
            for feature in features:
                feature = process_feat(feature, 32)  # ucf(32, 2048)
                divided_features.append(feature)

            divided_features = torch.stack(divided_features, dim=0)

            # MGFN
            divided_features = divided_features.unsqueeze(0)
            divided_features = divided_features.permute(0, 2, 1, 3).to(device)
            # _, _, _, _, _, _, _, logits, _, _, _ = model(divided_features) # Model2
            _, _, _, _, _, _, logits, _, _, _ = model(divided_features)  # Model
            logits = torch.squeeze(logits, 1)
            logits = torch.mean(logits, 0)
            print("iter:", it)
            print("logits")
            print(logits)
            print(logits.shape)
            # Publish logits
            msg = Float32MultiArray()
            msg.data = torch.squeeze(logits).cpu().detach().tolist()
            self.publisher.publish(msg)
            print("publish message")


def main(args):
    """Initializes and cleanup ros node"""
    ap = AccidentPrediction()
    rospy.init_node("mgfn")
    try:
        rospy.spin()

    except KeyboardInterrupt:
        print("Shutting down ROS Image module")


if __name__ == "__main__":
    main(sys.argv)
