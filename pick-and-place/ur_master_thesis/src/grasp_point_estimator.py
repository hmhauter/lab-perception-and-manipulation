import open3d as o3d
import numpy as np
import rospy

"""
Node for estimating the grasp point for the parallel gripper
Point is estimatedy registration. If that faills PCA is performed instead
Can be called with estimate_grasp_point
"""

class GraspPointEstimator:
    def __init__(self, width, length):
        o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)
        self.width = width
        self.length = length
        self.TOLERANCE = 2 / 100

    def __perform_pca(self, pcl_surface):

        mean, covariance_matrix = pcl_surface.compute_mean_and_covariance()
        eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)

        sorted_indices = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[sorted_indices]
        eigenvectors = eigenvectors[:, sorted_indices]

        principal_axes = eigenvectors[:, :3]

        lines = []
        for i in range(3):
            endpoint = mean + 0.5 * principal_axes[:, i]
            lines.append(np.vstack((mean, endpoint)))

        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(np.vstack(lines))
        line_set.lines = o3d.utility.Vector2iVector(
            np.array([[0, 1], [2, 3], [4, 5]])
        )  # Define line segments

        return eigenvalues, eigenvectors, mean, line_set

    def __create_pcl(self, rgb_image, depth_data, camera_intrinsics):
        color_image_o3d = o3d.geometry.Image(np.array(rgb_image))
        depth_image_o3d = o3d.geometry.Image(np.array(depth_data))
        fx = fy = camera_intrinsics["f"]
        cx = camera_intrinsics["cx"]
        cy = camera_intrinsics["cy"]

        rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color_image_o3d,
            depth_image_o3d,
            depth_scale=1000.0,
            depth_trunc=3.0,
            convert_rgb_to_intensity=False,
        )

        custom_intrinsic = o3d.camera.PinholeCameraIntrinsic(
            width=640, height=480, fx=fx, fy=fy, cx=cx, cy=cy
        )
        camera = o3d.camera.PinholeCameraIntrinsic(custom_intrinsic)

        point_cloud = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, camera)

        return point_cloud

    def __detect_suface(self, pcd):

        plane_model, inliers = pcd.segment_plane(
            distance_threshold=0.002, ransac_n=3, num_iterations=1000
        )

        [a, b, c, d] = plane_model

        inlier_points = pcd.select_by_index(inliers)
        inlier_points.paint_uniform_color([1, 0, 0])

        return inlier_points, plane_model

    def __reconstruct_surface(self, pcl_surface):
        alpha = 0.1  # lower apha means more detail
        num_points = 5000

        # Perform alpha shape reconstruction to create a mesh
        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(
            pcl_surface, alpha
        )
        pcl_surface_smooth = mesh.sample_points_uniformly(number_of_points=num_points)
        return pcl_surface_smooth

    def __calculate_grasp_points(self, eigenvectors, mean):

        gl_1 = mean + (self.length / 2) * eigenvectors[:, 0]
        gl_2 = mean - (self.length / 2) * eigenvectors[:, 0]

        gw_1 = mean + (self.width / 2) * eigenvectors[:, 1]
        gw_2 = mean - (self.width / 2) * eigenvectors[:, 1]

        p1 = (
            mean
            + (self.length / 2) * eigenvectors[:, 0]
            + (self.width / 2) * eigenvectors[:, 1]
        )
        p2 = (
            mean
            + (self.length / 2) * eigenvectors[:, 0]
            - (self.width / 2) * eigenvectors[:, 1]
        )
        return gl_1, gl_2, gw_1, gw_2, p1, p2

    def __get_bbox(self, pcl_surface_smooth):
        # helps to assess the quality of the created pointcloud
        obb = o3d.geometry.OrientedBoundingBox.create_from_points(
            pcl_surface_smooth.points
        )

        obb_dimensions = obb.get_max_bound() - obb.get_min_bound()

        width = max(obb_dimensions)

        length = sorted(obb_dimensions)[-2]

        return obb, length, width

    def __get_pcl_from_bbox(self, oriented_bbox):

        center = np.array(oriented_bbox.center)
        extent = np.array(oriented_bbox.extent)
        rotation_matrix = np.array(oriented_bbox.R)

        num_points = 5000

        points = np.random.uniform(-0.5, 0.5, (num_points, 3))

        points = points * extent

        points = np.dot(points, rotation_matrix.T)

        points = points + center

        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(points)

        return point_cloud

    def apply_transformation(self, T, OJ_EIGENVECTORS, OJ_MEAN):

        OJ_MEAN_H = np.append(OJ_MEAN, 1)

        transformed_OJ_MEAN_H = np.dot(T, OJ_MEAN_H)

        transformed_OJ_MEAN = transformed_OJ_MEAN_H[:3] / transformed_OJ_MEAN_H[3]

        R = T[:3, :3]

        transformed_OJ_EIGENVECTORS = np.dot(R, OJ_EIGENVECTORS)

        return transformed_OJ_EIGENVECTORS, transformed_OJ_MEAN

    def compute_rotation_matrix(self, eigenvecs1, eigenvecs2):

        eigenvecs1 = eigenvecs1 / np.linalg.norm(eigenvecs1, axis=0)
        eigenvecs2 = eigenvecs2 / np.linalg.norm(eigenvecs2, axis=0)

        R = np.dot(eigenvecs1, np.linalg.inv(eigenvecs2))

        return R

    def compute_translation_vector(self, origin1, origin2, R):
        t = origin1 - np.dot(R, origin2)
        return t

    def create_line_set(self, eigv, mean):

        principal_axes = eigv[:, :3]
        lines = []
        for i in range(3):
            endpoint = mean + 0.5 * principal_axes[:, i]
            lines.append(np.vstack((mean, endpoint)))

        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(np.vstack(lines))
        line_set.lines = o3d.utility.Vector2iVector(np.array([[0, 1], [2, 3], [4, 5]]))
        return line_set

    def normalize_rows(self, matrix):

        row_norms = np.linalg.norm(matrix, axis=1, keepdims=True)

        normalized_matrix = matrix / row_norms

        return normalized_matrix

    def register_pcl(self, camera_pointcloud, mean, eigenvectors):
        OJ_EIGENVECTORS = np.array(
            [
                [-0.97345, -0.22784, -0.022123],
                [-0.22528, 0.93636, 0.26923],
                [0.040626, -0.26707, 0.96282],
            ]
        )
        OJ_EIGENVECTORS = self.normalize_rows(OJ_EIGENVECTORS)
        OJ_MEAN = np.array([-0.031579, -0.066232, -0.44723])
        initial_transformation = np.eye(4)
        R_new = self.compute_rotation_matrix(eigenvectors, OJ_EIGENVECTORS)
        initial_transformation[:3, 3] = self.compute_translation_vector(
            mean, OJ_MEAN, R_new
        )
        initial_transformation[:3, :3] = R_new

        threshold = 0.01
        import copy

        ojs_pointcloud = o3d.io.read_point_cloud("/transformed_source.ply")
        ojs_pointcloud_copy = copy.deepcopy(ojs_pointcloud)
        line_set = self.create_line_set(OJ_EIGENVECTORS, OJ_MEAN)

        ojs_pointcloud.transform(initial_transformation)
        transformed_eigenvecs1, transformed_origin1 = self.apply_transformation(
            initial_transformation, OJ_EIGENVECTORS, OJ_MEAN
        )

        transformed_eigenvecs1 = self.normalize_rows(transformed_eigenvecs1)
        line_set1 = self.create_line_set(transformed_eigenvecs1, transformed_origin1)

        pink_color = [255 / 255.0, 192 / 255.0, 203 / 255.0]

        ojs_pointcloud.colors = o3d.utility.Vector3dVector(pink_color)
        o3d.visualization.draw_geometries([ojs_pointcloud, camera_pointcloud])

        voxel_size = 0.005

        ojs_pointcloud.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
        )
        camera_pointcloud.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
        )

        eye = np.eye(4)

        threshold = 2 * voxel_size
        icp_result = o3d.pipelines.registration.registration_icp(
            ojs_pointcloud,
            camera_pointcloud,
            threshold,
            eye,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        )

        registered_pointcloud = ojs_pointcloud.transform(icp_result.transformation)

        post_icp_evaluation = o3d.pipelines.registration.evaluate_registration(
            registered_pointcloud, camera_pointcloud, threshold, np.identity(4)
        )

        fitness = post_icp_evaluation.fitness

        transformed_eigenvecs3, transformed_origin3 = self.apply_transformation(
            icp_result.transformation, transformed_eigenvecs1, transformed_origin1
        )
        transformed_eigenvecs3 = self.normalize_rows(transformed_eigenvecs3)

        principal_axes = transformed_eigenvecs3[:, :3]

        lines = []
        for i in range(3):
            endpoint = transformed_origin3 + 0.5 * principal_axes[:, i]
            lines.append(np.vstack((transformed_origin3, endpoint)))

        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(np.vstack(lines))
        line_set.lines = o3d.utility.Vector2iVector(np.array([[0, 1], [2, 3], [4, 5]]))

        o3d.visualization.draw_geometries([registered_pointcloud, camera_pointcloud])

        return (
            transformed_eigenvecs3,
            transformed_origin3,
            fitness,
            registered_pointcloud,
            camera_pointcloud,
            line_set,
        )

    def estimate_grasp_point(self, rgb_image, depth_data, camera_intrinsics):
        pcl = self.__create_pcl(rgb_image, depth_data, camera_intrinsics)
        T = [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]]
        pcl = pcl.transform(T)
        pcl_surface, plane_model = self.__detect_suface(pcl)
        pcl_surface_smooth = self.__reconstruct_surface(pcl_surface)

        colors = np.zeros((len(pcl_surface_smooth.points), 3))
        colors[: len(pcl_surface_smooth.points)] = [255, 255, 0]

        pcl_surface_smooth.colors = o3d.utility.Vector3dVector(colors)

        bbox, bbox_length, bbox_width = self.__get_bbox(pcl_surface_smooth)

        eigenvalues, eigenvectors, mean, line_set = self.__perform_pca(
            pcl_surface_smooth
        )

        gl_1, gl_2, gw_1, gw_2, p1, p2 = self.__calculate_grasp_points(
            eigenvectors, mean
        )
        grasping_points = np.array([mean, gl_1, gl_2, gw_1, gw_2])
        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(grasping_points)

        colors = np.zeros((len(grasping_points), 3))
        colors[: len(grasping_points)] = [0, 255, 0]

        point_cloud.colors = o3d.utility.Vector3dVector(colors)

        mesh_coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=0.1, origin=[0, 0, 0]
        )

        (
            transformed_eigenvecs3,
            transformed_origin3,
            fitness,
            registered_pointcloud,
            registered_pcl,
            registered_line_set,
        ) = self.register_pcl(pcl, mean, eigenvectors)

        if fitness > 0.9:

            gl_1, gl_2, gw_1, gw_2, p1, p2 = self.__calculate_grasp_points(
                transformed_eigenvecs3, transformed_origin3
            )
            grasping_points = np.array([transformed_origin3, gl_1, gl_2, gw_1, gw_2])
            point_cloud = o3d.geometry.PointCloud()
            point_cloud.points = o3d.utility.Vector3dVector(grasping_points)
            return (
                transformed_eigenvecs3,
                transformed_origin3,
                gl_1,
                gl_2,
                gw_1,
                gw_2,
                point_cloud,
                registered_line_set,
                registered_pcl,
                mesh_coordinate_frame,
                registered_pointcloud,
                fitness,
                pcl_surface,
                pcl_surface_smooth,
            )

        rospy.logerr("Registration failed: Use PCA instead...")
        return (
            eigenvectors,
            mean,
            gl_1,
            gl_2,
            gw_1,
            gw_2,
            point_cloud,
            line_set,
            pcl,
            mesh_coordinate_frame,
            registered_pointcloud,
            fitness,
            pcl_surface,
            pcl_surface_smooth,
        )
