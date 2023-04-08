#!/usr/bin/env python3

# Integration with ROS2 for testing in gazebo simulation

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo, PointCloud2
from cv_bridge import CvBridge, CvBridgeError
import cv2

# import open3d as o3d
import numpy as np

# import imutils


class imageSubscriber(Node):
    def __init__(self):

        super().__init__('image_subscriber')

        # self.contour_depth = np.zeros([720,1280], np.uint16)

        self.color_sub = self.create_subscription(
            Image, '/camera/color/image_raw', self.color_callback, 10
        )

        self.depth_sub = self.create_subscription(
            Image, 'camera/depth/image_rect_raw', self.depth_callback, 10
        )

        self.info_sub = self.create_subscription(
            CameraInfo, '/camera/depth/camera_info', self.info_callback, 10
        )

        self.br = CvBridge()
        self.camera_info_pub = self.create_publisher(
            CameraInfo, '/cov_info', 10
        )
        self.depth_publisher = self.create_publisher(Image, 'output_depth', 10)

        # self.new_depth_publisher = self.create_publisher(
        #     Image,
        #     'new_output_depth',
        #     10
        # )

        # self.pc_publisher = self.create_publisher(
        #     PointCloud2,
        #     'output_pointcloud',
        #     10
        # )

        self.timer = self.create_timer(
            0.1, self.timer_callback  # publishing every 0.1 second
        )

        self.contour_depth = np.zeros((720, 1280), np.uint16)
        # self.new_output_depth_image = np.zeros((720,1280), np.uint16)

        self.depth_image = np.zeros((720, 1280), np.uint16)
        # self.rgbd_image = None

        self.pcd = None

    def info_callback(self, data):
        self.get_logger().info('Receiving camera info')
        self.camera_info = data
        # self.camera_info.header.stamp = self.get_clock().now().to_msg()
        # self.camera_info_pub.publish(self.camera_info)

    def timer_callback(self):
        try:
            depth_image = self.br.cv2_to_imgmsg(self.contour_depth)
            time = self.get_clock.now().to_msg()
            depth_image.header.stamp = time
            self.camera_info.header.stamp = time
            self.depth_publisher.publish(depth_image)
            self.camera_info_pub.publish(self.camera_info)
        except:
            pass
        # self.new_depth_publisher.publish(
        #     self.br.cv2_to_imgmsg(self.contour_depth)
        # )
        # self.pc_publisher.publish(
        #     self.pcd
        # )

    def depth_callback(self, data):
        self.get_logger().info('Receiving depth frame')
        self.depth_image = self.br.imgmsg_to_cv2(data, 'passthrough')
        # cv2.imshow("original depth",self.depth_image)
        # cv2.waitKey(1)

    def color_callback(self, data):
        self.get_logger().info('Receiving color frame')

        self.color_image = self.br.imgmsg_to_cv2(data, 'bgr8')

        self.numpy_color_image = np.array(
            self.color_image
        )     # used in pcd_processing

        # --------------------------- Perspective Transformation --------------------------------------------------------

        # self.pts2 = np.float32([[640, 180], [1280, 180],
        #                    [0, 1080], [1920, 1080]])

        # self.pts1 = np.float32([[0, 0], [1920, 0],
        #                    [0, 1080], [1920, 1080]])

        # self.transformation_matrix = cv2.getPerspectiveTransform(self.pts1, self.pts2)

        # self.transform_result = cv2.warpPerspective(self.gray, self.transformation_matrix, (1920, 1080))

        # --------------------------------------------------------------------------------------------------------------

        # --------------------------------- Color Filtering -------------------------------------------------------------

        self.hsv = cv2.cvtColor(self.color_image, cv2.COLOR_RGB2HSV)

        self.lower_white_hsv = (0, 0, 168)         # 0,0,200          0,0,168
        self.upper_white_hsv = (
            172,
            111,
            255,
        )     # 145,60,255       172,111,255

        self.hsv_color_mask = cv2.inRange(
            self.hsv, self.lower_white_hsv, self.upper_white_hsv
        )
        # self.hsv_mask = cv2.bitwise_and(self.color_image, self.color_image, mask = self.hsv_color_mask)

        # cv2.imshow("hsv", self.hsv_color_mask)

        # --------------------------------------------------------------------------------------------------------------

        # --------------------------------- rectangular region of interest ------------------------------------------------

        self.mask = np.zeros(self.hsv_color_mask.shape[:2], dtype='uint8')
        cv2.rectangle(self.mask, (0, 300), (1280, 720), (255, 0, 0), -1)

        self.masked_image = cv2.bitwise_and(
            self.hsv_color_mask, self.mask
        )     # total mask, HSV + ROI

        cv2.imshow('masked_image', self.masked_image)

        # -----------------------------------------------------------------------------------------------------------------

        # --------------------------------- image processing ---------------------------------------------------------------

        # applying a gaussian blur to the grayscale image
        self.blur = cv2.GaussianBlur(self.masked_image, (3, 3), 0)

        # detecting edges in the image using canny
        self.edges = cv2.Canny(self.blur, 50, 150)

        # define a (3, 3) structuring element
        self.kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

        # apply the dilation operation to the edged image
        self.dilated = cv2.dilate(self.edges, self.kernel, iterations=1)

        # find the contours in the edged image
        self.contours, self.hierarchy = cv2.findContours(
            self.dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        # cnts = imutils.grab_contours(contours)
        self.cnts = sorted(self.contours, key=cv2.contourArea, reverse=True)[
            :1
        ]

        self.image_copy = self.color_image.copy()

        # self.approx = cv2.approxPolyDP(self.cnts, 0.01* cv2.arcLength(self.cnts, True), True)
        # if len(self.approx) >= 8 :
        #     cv2.drawContours(self.image_copy, [self.approx], 0, (255, 0, 0), 5)

        # self.contour_depth = np.zeros((240,320))      # Defined in the class constructor, hence not required here

        # ----------------------------------- bitwise_and on depth image with pothole mask ---------------------------

        self.potholes = np.zeros((720, 1280), dtype='uint8')
        self.contour_depth = np.zeros((720, 1280), np.uint16)

        for contour in self.cnts:

            approx = cv2.approxPolyDP(
                contour, 0.01 * cv2.arcLength(contour, True), True
            )
            # cv2.drawContours(self.image_copy, [approx], 0, (0, 0, 0), 5)
            if len(approx) >= 8:
                cv2.drawContours(self.image_copy, [approx], 0, (255, 0, 0), 5)
                # In this case, "contour" is an array of points.
                # If I loop over every point in the coordinate array, this will become
                # very inefficient.

                self.potholes = cv2.fillPoly(
                    self.potholes, pts=[contour], color=(255, 255, 255)
                )

                # self.new_output_depth_image = cv2.bitwise_and(self.depth_image,self.depth_image,mask=self.potholes)
                print('Pothole Found\n')
                # print(self.new_output_depth_image.dtype,"\n",self.potholes.dtype)
                # print("\n", type(self.new_output_depth_image), "\n", type(self.potholes))

                self.contour_depth = cv2.bitwise_and(
                    self.depth_image, self.depth_image, mask=self.potholes
                )
                print('I also work !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')

                # No need for this slowass algorithm to get depth, just do bitwise_and :)

                # for i in contour:           # Access each element in the depth image
                #     # np.append(              # and then store it to the corrosponding location in the np.array
                #     #     self.contour_depth,
                #     #     (np.flip(i)).flatten()
                #     # )
                #     depth_coordinate = (np.flip(i)).flatten()
                #     depth_x = depth_coordinate[0]
                #     depth_y = depth_coordinate[1]
                #     # print(depth_coordinate)
                #     # print(depth_x, "\t", depth_y)
                #     # depth_data_xy = self.depth_image[depth_x][depth_y]

                #     # print(depth_data_xy,"\n")

                #     # self.contour_depth[depth_x][depth_y] = depth_data_xy
                #     # print(type(self.depth_image))
                #     try:
                #         self.contour_depth[depth_x][depth_y] = self.depth_image[depth_x][depth_y]
                #     except:
                #         continue
                # print(self.contour_depth,"\n")

                # print((np.flip(i)).flatten(),"\t hello \t")

                # self.pcd_processing()
                # print(self.contour_depth,"\n")
                # print(contour)

        cv2.imshow('potholes', self.potholes)
        cv2.waitKey(1)
        # self.depth_publisher.publish(
        #     self.br.cv2_to_imgmsg(self.contour_depth)
        # )
        # # self.processing(self.color_image)

    # ------------------------------------------- Point Cloud Processing --------------------------------

    # def pcd_processing(self):

    #     self.output_depth_image = o3d.geometry.Image((self.contour_depth).astype(np.uint16))
    #     self.output_color_image = o3d.geometry.Image((self.numpy_color_image).astype(np.uint8))

    #     print(self.output_color_image)

    #     print(self.output_depth_image)
    #     print("\n")

    #     self.rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
    #         self.output_color_image,
    #         self.output_depth_image # Change this original depth image to only the pothole depth.
    #     )

    #     # print(self.rgbd_image, "\n")

    #     self.pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
    #         self.rgbd_image,
    #         o3d.camera.PinholeCameraIntrinsic(
    #             height = 240,
    #             width = 320,
    #             fx = self.camera_info[0],
    #             fy = self.camera_info[4],
    #             cx = self.camera_info[2],
    #             cy = self.camera_info[5]
    #         )
    #     )

    #     # print(self.pcd)

    #     self.pcd.transform(
    #         [
    #             [1, 0, 0, 0],
    #             [0, -1, 0, 0],
    #             [0, 0, -1, 0],
    #             [0, 0, 0, 1]
    #         ]
    #     )

    #     print(self.pcd)
    #     print()
    #     print(type(self.pcd))

    # ----------------------------------------------------------------------------------------------------------------


def main(args=None):

    rclpy.init(args=args)

    image_subscriber = imageSubscriber()

    rclpy.spin(image_subscriber)


if __name__ == '__main__':
    main()
