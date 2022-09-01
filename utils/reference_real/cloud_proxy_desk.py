import time

import rospy
from sensor_msgs.msg import Image
from sensor_msgs import point_cloud2
from sensor_msgs.msg import PointCloud2
import ros_numpy
from sklearn.impute import SimpleImputer
import numpy as np
# import cupy as np
import scipy
import matplotlib.pyplot as plt
import tf2_ros
import tf
from src.utils import transformation
import time
import skimage.transform
# import scipy
import open3d

class CloudProxyDesk:
    def __init__(self, desk_center=[-0.527, -0.005], z_min=-0.07, enable_cam3=True):
        self.topic1 = '/camera/depth/points'
        self.sub1 = rospy.Subscriber(self.topic1, PointCloud2, self.callbackCloud1, queue_size=1)
        self.msg1 = None
        self.cloud1 = None

        self.topic2 = '/k4a/points2'
        self.sub2 = rospy.Subscriber(self.topic2, PointCloud2, self.callbackCloud2, queue_size=1)
        self.msg2 = None
        self.cloud2 = None

        self.topic3 = '/cam1/depth/points'
        self.sub3 = rospy.Subscriber(self.topic3, PointCloud2, self.callbackCloud3, queue_size=1)
        self.msg3  = None
        self.cloud3 = None

        self.tfBuffer = tf2_ros.Buffer()
        self.tfListener = tf2_ros.TransformListener(self.tfBuffer)

        self.image = None
        self.cloud = None

        self.z_min = z_min
        self.desk_center = desk_center
        self.enable_cam3 = enable_cam3

    def callbackCloud1(self, msg):
        self.msg1 = msg
        cloudTime = self.msg1.header.stamp
        cloudFrame = self.msg1.header.frame_id
        # cloud = np.array(list(point_cloud2.read_points(self.msg)))[:, 0:3]
        cloud = ros_numpy.point_cloud2.pointcloud2_to_xyz_array(self.msg1, remove_nans=True)
        # pc = ros_numpy.numpify(self.msg1)
        # height = pc.shape[0]
        # width = pc.shape[1]
        # cloud = np.zeros((height * width, 3), dtype=np.float32)
        # cloud[:, 0] = np.resize(pc['x'], height * width)
        # cloud[:, 1] = np.resize(pc['y'], height * width)
        # cloud[:, 2] = np.resize(pc['z'], height * width)
        mask = np.logical_not(np.isnan(cloud).any(axis=1))
        cloud = cloud[mask]
        # print("Received Structure cloud with {} points.".format(cloud.shape[0]))
        T = self.lookupTransform(cloudFrame, 'base', rospy.Time(0))
        cloud = self.transform(cloud, T)
        self.cloud1 = cloud

    def callbackCloud2(self, msg):
        self.msg2 = msg
        cloudTime = self.msg2.header.stamp
        cloudFrame = self.msg2.header.frame_id
        # cloud = np.array(list(point_cloud2.read_points(self.msg2, field_names = ("x", "y", "z"), skip_nans=True)))[:, 0:3]
        cloud = ros_numpy.point_cloud2.pointcloud2_to_xyz_array(self.msg2, remove_nans=True)
        # pc = ros_numpy.numpify(self.msg2)
        # height = pc.shape[0]
        # width = pc.shape[1]
        # cloud = np.zeros((height * width, 3), dtype=np.float32)
        # cloud[:, 0] = np.resize(pc['x'], height * width)
        # cloud[:, 1] = np.resize(pc['y'], height * width)
        # cloud[:, 2] = np.resize(pc['z'], height * width)
        mask = np.logical_not(np.isnan(cloud).any(axis=1))
        cloud = cloud[mask]
        # print("Received Structure cloud with {} points.".format(cloud.shape[0]))
        T = self.lookupTransform(cloudFrame, 'base', rospy.Time(0))
        cloud = self.transform(cloud, T)
        self.cloud2 = cloud

    def callbackCloud3(self, msg):
        self.msg3 = msg
        cloudTime = self.msg3.header.stamp
        cloudFrame = self.msg3.header.frame_id
        # cloud = np.array(list(point_cloud2.read_points(self.msg2, field_names = ("x", "y", "z"), skip_nans=True)))[:, 0:3]
        cloud = ros_numpy.point_cloud2.pointcloud2_to_xyz_array(self.msg3, remove_nans=True)
        # pc = ros_numpy.numpify(self.msg2)
        # height = pc.shape[0]
        # width = pc.shape[1]
        # cloud = np.zeros((height * width, 3), dtype=np.float32)
        # cloud[:, 0] = np.resize(pc['x'], height * width)
        # cloud[:, 1] = np.resize(pc['y'], height * width)
        # cloud[:, 2] = np.resize(pc['z'], height * width)
        mask = np.logical_not(np.isnan(cloud).any(axis=1))
        cloud = cloud[mask]
        # print("Received Structure cloud with {} points.".format(cloud.shape[0]))
        T = self.lookupTransform(cloudFrame, 'base', rospy.Time(0))
        cloud = self.transform(cloud, T)
        self.cloud3 = cloud

    def transform(self, cloud, T, isPosition=True):
        '''Apply the homogeneous transform T to the point cloud. Use isPosition=False if transforming unit vectors.'''

        n = cloud.shape[0]
        cloud = cloud.T
        augment = np.ones((1, n)) if isPosition else np.zeros((1, n))
        cloud = np.concatenate((cloud, augment), axis=0)
        cloud = np.dot(T, cloud)
        cloud = cloud[0:3, :].T
        return cloud

    def lookupTransform(self, fromFrame, toFrame, lookupTime=rospy.Time(0)):
        """
        Lookup a transform in the TF tree.
        :param fromFrame: the frame from which the transform is calculated
        :type fromFrame: string
        :param toFrame: the frame to which the transform is calculated
        :type toFrame: string
        :return: transformation matrix from fromFrame to toFrame
        :rtype: 4x4 np.array
        """

        transformMsg = self.tfBuffer.lookup_transform(toFrame, fromFrame, lookupTime, rospy.Duration(1.0))
        translation = transformMsg.transform.translation
        pos = [translation.x, translation.y, translation.z]
        rotation = transformMsg.transform.rotation
        quat = [rotation.x, rotation.y, rotation.z, rotation.w]
        T = tf.transformations.quaternion_matrix(quat)
        T[0:3, 3] = pos
        return T

    def interpolate(self, depth):
        """
        Fill nans in depth image
        """
        # a boolean array of (width, height) which False where there are missing values and True where there are valid (non-missing) values
        mask = np.logical_not(np.isnan(depth))
        # array of (number of points, 2) containing the x,y coordinates of the valid values only
        xx, yy = np.meshgrid(np.arange(depth.shape[1]), np.arange(depth.shape[0]))
        xym = np.vstack((np.ravel(xx[mask]), np.ravel(yy[mask]))).T

        # the valid values in the first, second, third color channel,  as 1D arrays (in the same order as their coordinates in xym)
        data0 = np.ravel(depth[:, :][mask])

        # three separate interpolators for the separate color channels
        interp0 = scipy.interpolate.NearestNDInterpolator(xym, data0)

        # interpolate the whole image, one color channel at a time
        result0 = interp0(np.ravel(xx), np.ravel(yy)).reshape(xx.shape)

        return result0

    def getNewPointCloud(self, gripper_pos):
        """
        get new point cloud, set self.cloud
        """
        start_time = time.time()
        while self.cloud1 is None or self.cloud2 is None or self.cloud3 is None:
            rospy.sleep(0.1)
        cloud1 = self.cloud1
        cloud2 = self.cloud2
        if self.enable_cam3:
            cloud3 = self.cloud3
        else:
            cloud3 = cloud2[:2]
        cloud = np.concatenate((cloud1, cloud2, cloud3))
        # filter workspace
        # cloud = cloud[(cloud[:, 2] > -0.2) * (cloud[:, 0] < -0.23) * (cloud[:, 0] > -0.8) * (np.abs(cloud[:, 1]) < 0.4)]
        # filter out arm and gripper
        # cloud = cloud[cloud[:, 2] < gripper_pos[2] + 0.05]
        # cloud = cloud[np.logical_not((cloud[:, 2] > max(gripper_pos[2], 0.05)) * (np.abs(cloud[:, 0] - gripper_pos[0]) < 0.08) * (np.abs(cloud[:, 1] - gripper_pos[1]) < 0.08))]
        half_size = 0.2
        long_extent = 0.2
        cloud = cloud[(cloud[:, 2] < max(gripper_pos[2], self.z_min + 0.05))]
        # filter ws x
        cloud = cloud[(cloud[:, 0] < self.desk_center[0] + long_extent) * (cloud[:, 0] > self.desk_center[0] - long_extent)]
        # filter ws y
        cloud = cloud[((cloud[:, 1] < self.desk_center[1] + half_size) * (cloud[:, 1] > self.desk_center[1] - half_size))]
        # filter ws z
        cloud = cloud[cloud[:, 2] > - 0.10]

        pcd = open3d.geometry.PointCloud()
        pcd.points = open3d.utility.Vector3dVector(cloud)
        cl, ind = pcd.remove_statistical_outlier(nb_neighbors=50, std_ratio=2.0)
        cloud = np.asarray(cl.points)

        # generate 'fake' point cloud for area outside the bins
        x = np.arange(self.desk_center[0]*1000-400, self.desk_center[0]*1000+400, 2)
        y = np.arange(self.desk_center[1]*1000-400, self.desk_center[1]*1000+400, 2)
        xx, yy = np.meshgrid(x, y)
        xx = xx/1000
        yy = yy/1000
        xx = xx.reshape(-1, 1)
        yy = yy.reshape(-1, 1)
        pts = np.concatenate([xx, yy, np.ones_like(yy)*(self.z_min-0.02)], 1)
        pts = pts[np.logical_not(((pts[:, 0] < self.desk_center[0] + half_size) * (pts[:, 0] > self.desk_center[0] - half_size) * (pts[:, 1] < self.desk_center[1] + half_size) * (pts[:, 1] > self.desk_center[1] - half_size)))]
        # pts = pts[np.logical_not(((pts[:, 1] < 0.239 + half_size) * (pts[:, 1] > 0.239 - half_size)) + ((pts[:, 1] < -0.21 + half_size) * (pts[:, 1] > -0.21 - half_size)))]
        cloud = np.concatenate([cloud, pts])
        # uncomment for visualization
        # pcd = open3d.geometry.PointCloud()
        # pcd.points = open3d.utility.Vector3dVector(cloud)
        # open3d.visualization.draw_geometries([pcd])
        self.cloud = cloud

        # def display_inlier_outlier(cloud, ind):
        #     inlier_cloud = cloud.select_by_index(ind)
        #     outlier_cloud = cloud.select_by_index(ind, invert=True)
        #
        #     print("Showing outliers (red) and inliers (gray): ")
        #     outlier_cloud.paint_uniform_color([1, 0, 0])
        #     inlier_cloud.paint_uniform_color([0.8, 0.8, 0.8])
        #     open3d.visualization.draw_geometries([inlier_cloud, outlier_cloud])

        # cl, ind = pcd.remove_statistical_outlier(nb_neighbors=50, std_ratio=2.0)
        # display_inlier_outlier(pcd, ind)
        # print(1)

    def clearPointCloud(self):
        self.cloud = None

    def getProjectImg(self, target_size, img_size, gripper_pos=(-0.5, 0, 0.1)):
        """
        return orthographic projection depth img from self.cloud
        target_size: img coverage size in meters
        img_size: img pixel size
        gripper_pos: the pos of the camera
        return depth image
        """
        if self.cloud is None:
            self.getNewPointCloud(gripper_pos)
        cloud = np.copy(self.cloud)
        cloud = cloud[(cloud[:, 2] < max(gripper_pos[2], self.z_min + 0.05))]
        view_matrix = transformation.euler_matrix(0, np.pi, 0).dot(np.eye(4))
        # view_matrix = np.eye(4)
        view_matrix[:3, 3] = [gripper_pos[0], -gripper_pos[1], gripper_pos[2]]
        view_matrix = transformation.euler_matrix(0, 0, -np.pi/2).dot(view_matrix)
        augment = np.ones((1, cloud.shape[0]))
        pts = np.concatenate((cloud.T, augment), axis=0)
        projection_matrix = np.array([
            [1 / (target_size / 2), 0, 0, 0],
            [0, 1 / (target_size / 2), 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
        tran_world_pix = np.matmul(projection_matrix, view_matrix)
        pts = np.matmul(tran_world_pix, pts)
        # pts[1] = -pts[1]
        pts[0] = (pts[0] + 1) * img_size / 2
        pts[1] = (pts[1] + 1) * img_size / 2

        pts[0] = np.round_(pts[0])
        pts[1] = np.round_(pts[1])
        mask = (pts[0] >= 0) * (pts[0] < img_size) * (pts[1] > 0) * (pts[1] < img_size)
        pts = pts[:, mask]
        # dense pixel index
        mix_xy = (pts[1].astype(int) * img_size + pts[0].astype(int))
        # lexsort point cloud first on dense pixel index, then on z value
        ind = np.lexsort(np.stack((pts[2], mix_xy)))
        # bin count the points that belongs to each pixel
        bincount = np.bincount(mix_xy)
        # cumulative sum of the bin count. the result indicates the cumulative sum of number of points for all previous pixels
        cumsum = np.cumsum(bincount)
        # rolling the cumsum gives the ind of the first point that belongs to each pixel.
        # because of the lexsort, the first point has the smallest z value
        cumsum = np.roll(cumsum, 1)
        cumsum[0] = bincount[0]
        cumsum[cumsum == np.roll(cumsum, -1)] = 0
        # pad for unobserved pixels
        cumsum = np.concatenate((cumsum, -1 * np.ones(img_size * img_size - cumsum.shape[0]))).astype(int)

        depth = pts[2][ind][cumsum]
        depth[cumsum == 0] = np.nan
        depth = depth.reshape(img_size, img_size)
        # fill nans
        depth = self.interpolate(depth)
        # mask = np.isnan(depth)
        # depth[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), depth[~mask])
        # imputer = SimpleImputer(missing_values=np.nan, strategy='median')
        # imputer_depth = imputer.fit_transform(depth)
        # if imputer_depth.shape != depth.shape:
        #     mask = np.isnan(depth)
        #     depth[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), depth[~mask])
        # else:
        #     depth = imputer_depth
        return depth

def main():
    rospy.init_node('test')
    cloudProxy = CloudProxyDesk()
    while True:
        cloudProxy.cloud = None
        obs = cloudProxy.getProjectImg(0.3, 128, (-0.525, 0.239, 0.13))
        # obs = -obs
        # obs -= obs.min()
        # obs = skimage.transform.resize(obs, (90, 90))
        plt.imshow(obs)
        plt.colorbar()
        plt.show()
        pass


if __name__ == '__main__':
    main()