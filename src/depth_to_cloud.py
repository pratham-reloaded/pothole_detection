#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile
from sensor_msgs.msg import Image, PointCloud2
from cv_bridge import CvBridge

# import numpy as np
import struct


class DepthToPointcloudNode(Node):
    def __init__(self):
        super().__init__('depth_to_pointcloud_node')

        # Initialize CvBridge
        self.bridge = CvBridge()

        # Subscribe to depth image topic
        self.subscriber = self.create_subscription(
            Image,
            '/output_depth',
            self.callback_depth_image,
            QoSProfile(depth=10),
        )

        # Publish point cloud
        self.publisher = self.create_publisher(
            PointCloud2, 'point_cloud_topic', QoSProfile(depth=10)
        )

    def callback_depth_image(self, msg):
        # Convert depth image to numpy array
        depth_array = self.bridge.imgmsg_to_cv2(msg)

        # Get image dimensions
        height, width = depth_array.shape

        # Create point cloud message
        pointcloud_msg = PointCloud2()

        # Fill in header
        pointcloud_msg.header = msg.header
        pointcloud_msg.header.frame_id = 'camera_link'

        # Set point cloud fields
        pointcloud_msg.height = height
        pointcloud_msg.width = width
        pointcloud_msg.fields.append(
            PointField(name='x', offset=0, datatype=7, count=1)
        )
        pointcloud_msg.fields.append(
            PointField(name='y', offset=4, datatype=7, count=1)
        )
        pointcloud_msg.fields.append(
            PointField(name='z', offset=8, datatype=7, count=1)
        )
        # pointcloud_msg.fields.append(
        # PointField(name='rgb', offset=12, datatype=7, count=1)
        # )
        pointcloud_msg.point_step = 16
        pointcloud_msg.row_step = pointcloud_msg.point_step * width
        pointcloud_msg.is_dense = True

        # Fill in point cloud data
        for v in range(height):
            for u in range(width):
                z = depth_array[v, u] / 1000.0   # convert to meters
                x = (u - self.cx) * z / self.fx
                y = (v - self.cy) * z / self.fy
                # rgb = struct.unpack('I', struct.pack('BBBB', *msg.data[v*width + u*4 : v*width + u*4 + 4]))[0]
                pointcloud_msg.data += struct.pack('fff', x, y, z)

        # Publish point cloud
        print(pointcloud_msg.data)
        self.publisher.publish(pointcloud_msg)


def main(args=None):
    rclpy.init(args=args)

    depth_to_pointcloud_node = DepthToPointcloudNode()

    rclpy.spin(depth_to_pointcloud_node)

    # depth_to_pointcloud_node.destroy_node()
    # rclpy.shutdown()


if __name__ == '__main__':
    main()
