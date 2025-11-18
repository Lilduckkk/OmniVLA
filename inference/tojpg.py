#!/usr/bin/env python3
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import os

class ImageSaver:
    def __init__(self):
        self.bridge = CvBridge()
        self.save_dir = rospy.get_param('~save_dir', '/home/pcl/OmniVLA/inference')
        os.makedirs(self.save_dir, exist_ok=True)
        self.image_count = 0
        rospy.Subscriber('/camera/color/image_raw', Image, self.callback)
        rospy.loginfo(f"ImageSaver 节点已启动，保存目录: {self.save_dir}")

    def callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            # filename = os.path.join(self.save_dir, f'image_{self.image_count:06d}.jpg')
            filename = os.path.join(self.save_dir, f'goal_img.jpg')
            cv2.imwrite(filename, cv_image)
            rospy.loginfo(f"已保存图片: {filename}")
            self.image_count += 1
        except Exception as e:
            rospy.logerr(f"保存图片失败: {e}")

if __name__ == '__main__':
    rospy.init_node('image_saver')
    saver = ImageSaver()
    rospy.spin()