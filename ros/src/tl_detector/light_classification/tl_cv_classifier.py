from styx_msgs.msg import TrafficLight
import cv2
import numpy as np


class TLCVClassifier(object):

    def __init__(self):

        self.debug = False

        # define color ranges
        self.hsv_red_low_low = np.array((0, 100, 100), dtype="uint8")
        self.hsv_red_low_high = np.array((10, 255, 255), dtype="uint8")
        self.hsv_red_high_low = np.array((160, 100, 100), dtype="uint8")
        self.hsv_red_high_high = np.array((179, 255, 255), dtype="uint8")
        self.hsv_yellow_low = np.array((25, 100, 100), dtype="uint8")
        self.hsv_yellow_high = np.array((35, 255, 255), dtype="uint8")
        self.hsv_green_low = np.array((60, 100, 100), dtype="uint8")
        self.hsv_green_high = np.array((100, 255, 255), dtype="uint8")

    def get_classification(self, image):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """

        # convert bgr to hsv
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        min_num_pixels = 75

        # Look for red pixels
        mask_1 = cv2.inRange(hsv, self.hsv_red_low_low, self.hsv_red_low_high)
        mask_2 = cv2.inRange(hsv, self.hsv_red_high_low, self.hsv_red_high_high)
        if cv2.countNonZero(mask_1) + cv2.countNonZero(mask_2) > min_num_pixels:
            print('Traffic light is red')
            return TrafficLight.RED

        # And green
        mask = cv2.inRange(hsv, self.hsv_green_low, self.hsv_green_high)
        if cv2.countNonZero(mask) > min_num_pixels:
            print('Traffic light is green')
            return TrafficLight.GREEN

        # Yellow for last as it's the most unlikely
        mask = cv2.inRange(hsv, self.hsv_yellow_low, self.hsv_yellow_high)
        if cv2.countNonZero(mask) > min_num_pixels:
            print('Traffic light is yellow')
            return TrafficLight.YELLOW

        # Nothing detected so go for unknown
        print('Traffic light is unknown')
        return TrafficLight.UNKNOWN
