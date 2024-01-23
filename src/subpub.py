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
from collections import deque

# numpy and OpenCV
import numpy as np
import cv2

# Ros libraries
import roslib
import rospy

# Ros Messages
from sensor_msgs.msg import CompressedImage, Image
from accident_prediction.msg import ImageStream

# We do not use cv_bridge it does not support CompressedImage in python
from cv_bridge import CvBridge, CvBridgeError

VERBOSE = False


class image_feature:
    def __init__(self):
        """Initialize ros publisher, ros subscriber"""
        # topic where we publish
        self.image_pub = rospy.Publisher("image_stream", ImageStream)
        # self.bridge = CvBridge()
        self.data = deque(maxlen=50)
        # subscribed Topic
        self.subscriber = rospy.Subscriber(
            "/carla/ego_vehicle/rgb_front/image", Image, self.callback, queue_size=1
        )
        if VERBOSE:
            print("subscribed to /carla/ego_vehicle/rgb_front/images")

    def callback(self, ros_data):
        """Callback function of subscribed topic.
        Here images get converted and features detected"""
        if VERBOSE:
            print('received image of type: "%s"' % ros_data.format)

        #### direct conversion to CV2 ####
        # image_np = self.bridge.imgmsg_to_cv2(ros_data, "bgr8")
        self.data.append(ros_data)

        # cv2.imshow("cv_img", image_np)
        # cv2.waitKey(2)

        #### Create Iamge Stream ####
        msg = ImageStream()
        # msg.time = rospy.Time.now()
        msg.data = list(self.data)
        # Publish new image
        if len(self.data) >= 50:
            self.image_pub.publish(msg)
            for i in range(10):
                self.data.popleft()

        # self.subscriber.unregister()


def main(args):
    """Initializes and cleanup ros node"""
    ic = image_feature()
    rospy.init_node("img2stream")
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down ROS Image feature detector module")
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main(sys.argv)
