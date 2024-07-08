import open3d as o3d
import numpy as np
import rospy

class GraspPointEstimator:
    def __init__(self, width, length):
        o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)
        self.width = width 
        self.length = length 
        self.TOLERANCE = 2 / 100

    
    def __perform_pca(self, pcl_surface):

        mean, covariance_matrix = pcl_surface.compute_mean_and_covariance()
        # print(mean)
        # print(covariance_matrix)
        # centered_points = (pcl_surface.points - mean)
        
        # normalized_points = (pcl_surface.points - mean) @ np.linalg.pinv(np.linalg.cholesky(covariance_matrix).T) # np.sqrt(np.diag(covariance_matrix))
        # covariance_matrix_normalized = np.cov(normalized_points, rowvar=False)

        # Perform eigenvalue decomposition
        # eigenvalues_old, eigenvectors_old = np.linalg.eigh(covariance_matrix)
        eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)

        # Sort eigenvalues and eigenvectors in descending order
        sorted_indices = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[sorted_indices]
        eigenvectors = eigenvectors[:, sorted_indices]

        # Get the principal axes
        principal_axes = eigenvectors[:, :3]  # Extract the first three eigenvectors
        # principal_axes_old = eigenvectors_old[:, :3]
        # Define line segments along principal axes
        lines = []
        for i in range(3):
            endpoint = mean + 0.5 * principal_axes[:, i]  # Extend the line segment by a factor of 0.1
            lines.append(np.vstack((mean, endpoint)))

        # lines_old = []
        # for i in range(3):
        #     endpoint = mean + 0.5 * principal_axes_old[:, i]  # Extend the line segment by a factor of 0.1
        #     lines_old.append(np.vstack((mean, endpoint)))

        # Create line set geometry
        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(np.vstack(lines))
        line_set.lines = o3d.utility.Vector2iVector(np.array([[0, 1], [2, 3], [4, 5]]))  # Define line segments

        # line_set_old = o3d.geometry.LineSet()
        # line_set_old.points = o3d.utility.Vector3dVector(np.vstack(lines_old))
        # line_set_old.lines = o3d.utility.Vector2iVector(np.array([[0, 1], [2, 3], [4, 5]]))  # Define line segments
        # line_set_old.colors = o3d.utility.Vector3dVector(np.array([[1,0,0],[0,1,0],[0,0,1]]))
        # debug_pcd = o3d.geometry.PointCloud()
        # debug_pcd.points = o3d.utility.Vector3dVector(normalized_points)
        # # Visualize point cloud and principal components
        # o3d.visualization.draw_geometries([debug_pcd, line_set, line_set_old])
        # Print eigenvalues and eigenvectors
        # print("Eigenvalues:")
        # print(eigenvalues)
        # print("\nEigenvectors:")
        # print(eigenvectors)
        return eigenvalues, eigenvectors, mean, line_set
    
    def __create_pcl(self, rgb_image, depth_data, camera_intrinsics):
        color_image_o3d = o3d.geometry.Image(np.array(rgb_image))
        depth_image_o3d = o3d.geometry.Image(np.array(depth_data))
        fx = fy = camera_intrinsics['f']
        cx = camera_intrinsics['cx']
        cy = camera_intrinsics['cy']

        rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(color_image_o3d, depth_image_o3d, 
                                                                        depth_scale=1000.0, 
                                                                        depth_trunc=3.0, 
                                                                        convert_rgb_to_intensity=False)

        # Source pointcloud
        custom_intrinsic = o3d.camera.PinholeCameraIntrinsic(width=640, height=480, fx=fx, fy=fy, cx=cx, cy=cy)
        camera = o3d.camera.PinholeCameraIntrinsic(
            custom_intrinsic
        )

        point_cloud = o3d.geometry.PointCloud.create_from_rgbd_image(
            rgbd_image, camera)
        # o3d.visualization.draw_geometries([point_cloud])
        return point_cloud
    
    def __detect_suface(self, pcd):
        # Perform plane detection using RANSAC
        plane_model, inliers = pcd.segment_plane(
            distance_threshold=0.002, 
            ransac_n=3, 
            num_iterations=1000)

        # Extract plane parameters
        [a, b, c, d] = plane_model

        # Extract plane points
        inlier_points = pcd.select_by_index(inliers)
        inlier_points.paint_uniform_color([1,0,0])
        # Draw
        # o3d.visualization.draw_geometries([pcd, inlier_points])
        return inlier_points, plane_model
    
    def __reconstruct_surface(self, pcl_surface):
        alpha = 0.1  # lower apha means more detail
        num_points = 5000

        # Perform alpha shape reconstruction to create a mesh
        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcl_surface, alpha)
        pcl_surface_smooth = mesh.sample_points_uniformly(number_of_points=num_points)
        return pcl_surface_smooth

    def __calculate_grasp_points(self, eigenvectors, mean):
        # LENGTH 
        gl_1 = mean + (self.length / 2) * eigenvectors[: ,0]
        gl_2 = mean - (self.length / 2) * eigenvectors[: ,0]
        # WIDTH 
        gw_1 = mean + (self.width / 2) * eigenvectors[: ,1]
        gw_2 = mean - (self.width / 2) * eigenvectors[: ,1]
        # Calculate ROI for OCR
        p1 = mean + (self.length / 2) * eigenvectors[: ,0] + (self.width / 2) * eigenvectors[: ,1]
        p2 = mean + (self.length / 2) * eigenvectors[: ,0] - (self.width / 2) * eigenvectors[: ,1]
        return gl_1, gl_2, gw_1, gw_2, p1, p2

    def __get_bbox(self, pcl_surface_smooth):
        # helps to assess the quality of the created pointcloud 
        obb = o3d.geometry.OrientedBoundingBox.create_from_points(pcl_surface_smooth.points)

        # Get the dimensions of the bounding box
        obb_dimensions = obb.get_max_bound() - obb.get_min_bound()

        # Width corresponds to the maximum dimension of the bounding box
        width = max(obb_dimensions)

        # Length corresponds to the second maximum dimension of the bounding box
        length = sorted(obb_dimensions)[-2]
        # print("FROM BBOX")
        # print("Width:", width)
        # print("Length:", length)

        return obb, length, width
    
    def __get_pcl_from_bbox(self, oriented_bbox):
        print(oriented_bbox)
        # Extract information from the oriented bounding box
        center = np.array(oriented_bbox.center)
        extent = np.array(oriented_bbox.extent)
        rotation_matrix = np.array(oriented_bbox.R)
        
        # Generate random points uniformly within the bounding box
        num_points = 5000  # You can adjust this number as needed
        
        # Generate random points within the bounding box
        points = np.random.uniform(-0.5, 0.5, (num_points, 3))
        
        # Scale points by the extent
        points = points * extent
        
        # Rotate points based on the rotation matrix
        points = np.dot(points, rotation_matrix.T)
        
        # Translate points to the center of the bounding box
        points = points + center

        # Create a point cloud from the sampled points
        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(points)

        return point_cloud
    
    # def apply_transformation(self, T, eigenvecs, origin):
    #     # Extract rotation part (upper-left 3x3 submatrix) and translation part (top-right 3x1 vector)
    #     R = T[:3, :3]
    #     t = T[:3, 3]
        
    #     # Apply rotation to eigenvectors
    #     transformed_eigenvecs = np.dot(eigenvecs, R.T)
        
    #     # Convert origin to homogeneous coordinates
    #     origin_homogeneous = np.append(origin, 1)
        
    #     # Apply the full 4x4 transformation matrix to the origin
    #     transformed_origin_homogeneous = np.dot(T, origin_homogeneous)
        
    #     # Convert back to 3D coordinates by dividing by the w component
    #     transformed_origin = transformed_origin_homogeneous[:3] / transformed_origin_homogeneous[3]
        
    #     return transformed_eigenvecs, transformed_origin


    def apply_transformation(self, T, OJ_EIGENVECTORS, OJ_MEAN):
        # Convert OJ_MEAN to homogeneous coordinates
        OJ_MEAN_H = np.append(OJ_MEAN, 1)
        
        # Apply the transformation matrix to OJ_MEAN
        transformed_OJ_MEAN_H = np.dot(T, OJ_MEAN_H)
        
        # Convert back to 3D coordinates
        transformed_OJ_MEAN = transformed_OJ_MEAN_H[:3] / transformed_OJ_MEAN_H[3]
        
        # Extract the rotation part of the transformation matrix
        R = T[:3, :3]
        
        # Apply the rotation to the eigenvectors
        transformed_OJ_EIGENVECTORS = np.dot(R, OJ_EIGENVECTORS)
        
        return transformed_OJ_EIGENVECTORS, transformed_OJ_MEAN

    def compute_rotation_matrix(self, eigenvecs1, eigenvecs2):
        # Normalize eigenvectors
        eigenvecs1 = eigenvecs1 / np.linalg.norm(eigenvecs1, axis=0)
        eigenvecs2 = eigenvecs2 / np.linalg.norm(eigenvecs2, axis=0)
        
        # Compute rotation matrix R
        R = np.dot(eigenvecs1, np.linalg.inv(eigenvecs2))
        
        # Handle reflection case (if det(R) < 0)
        # if np.linalg.det(R) < 0:
        #     eigenvecs2[:, 0] *= -1  # Invert one axis to correct reflection
        #     R = np.dot(eigenvecs1.T, eigenvecs2)  # Recompute R
        return R
    def compute_translation_vector(self, origin1, origin2, R):
        t = origin1 - np.dot(R, origin2)
        return t

    def create_line_set(self, eigv, mean):
               
        ## DEBUG
        # Get the principal axes
        principal_axes = eigv[:, :3]  # Extract the first three eigenvectors
        # principal_axes_old = eigenvectors_old[:, :3]
        # Define line segments along principal axes
        lines = []
        for i in range(3):
            endpoint = mean + 0.5 * principal_axes[:, i]  # Extend the line segment by a factor of 0.1
            lines.append(np.vstack((mean, endpoint)))

        # Create line set geometry
        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(np.vstack(lines))
        line_set.lines = o3d.utility.Vector2iVector(np.array([[0, 1], [2, 3], [4, 5]]))  # Define line segments
        return line_set
    
    def normalize_rows(self, matrix):
        # Calculate the norm (magnitude) of each row
        row_norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        
        # Normalize each row by dividing by its norm
        normalized_matrix = matrix / row_norms
        
        return normalized_matrix
    

    def register_pcl(self, camera_pointcloud, mean, eigenvectors):
        OJ_EIGENVECTORS = np.array([[   -0.97345,    -0.22784,   -0.022123],
                                    [   -0.22528,     0.93636,     0.26923],
                                    [   0.040626,   -0.26707,     0.96282]])
        OJ_EIGENVECTORS = self.normalize_rows(OJ_EIGENVECTORS)
        OJ_MEAN = np.array([  -0.031579,   -0.066232,    -0.44723])
        initial_transformation = np.eye(4)      
        R_new = self.compute_rotation_matrix(eigenvectors, OJ_EIGENVECTORS)
        initial_transformation[:3, 3] = self.compute_translation_vector(mean, OJ_MEAN, R_new)
        initial_transformation[:3, :3] = R_new


        threshold = 0.01  # 2cm distance threshold
        # Load your point clouds
        import copy
        ojs_pointcloud = o3d.io.read_point_cloud("/home/apo/catkin_ws/src/ur_master_thesis/src/transformed_source.ply")
        ojs_pointcloud_copy = copy.deepcopy(ojs_pointcloud)
        line_set = self.create_line_set(OJ_EIGENVECTORS, OJ_MEAN)

        # ##############################
        ojs_pointcloud.transform(initial_transformation)
        transformed_eigenvecs1, transformed_origin1 = self.apply_transformation(initial_transformation, OJ_EIGENVECTORS, OJ_MEAN)

        transformed_eigenvecs1 = self.normalize_rows(transformed_eigenvecs1)
        line_set1 = self.create_line_set(transformed_eigenvecs1, transformed_origin1)
        # Visualize the registered point clouds after ICP refinement
        # o3d.visualization.draw_geometries([ojs_pointcloud, line_set1, ojs_pointcloud_copy, line_set])
        pink_color = [255 / 255.0, 192 / 255.0, 203 / 255.0]


        ojs_pointcloud.colors = o3d.utility.Vector3dVector(pink_color)
        o3d.visualization.draw_geometries([ojs_pointcloud, camera_pointcloud])
        
        # Downsample the point clouds
        voxel_size = 0.005

        # Compute normals for the original point clouds (needed for ICP)
        ojs_pointcloud.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
        camera_pointcloud.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
        print("PERFORM ICP")
        eye = np.eye(4)
        # Perform ICP refinement
        threshold = 2 * voxel_size 
        icp_result = o3d.pipelines.registration.registration_icp(
            ojs_pointcloud, camera_pointcloud, threshold, eye,
            o3d.pipelines.registration.TransformationEstimationPointToPoint()
        )

        # Apply the transformation to the original point cloud
        registered_pointcloud = ojs_pointcloud.transform(icp_result.transformation)

        post_icp_evaluation = o3d.pipelines.registration.evaluate_registration(registered_pointcloud, camera_pointcloud, threshold, np.identity(4))
        print("ICP Fitness")
        print(post_icp_evaluation.fitness)
        fitness = post_icp_evaluation.fitness

        print("ICP transformation matrix:")
        print(icp_result.transformation)

        
        
        transformed_eigenvecs3, transformed_origin3 = self.apply_transformation(icp_result.transformation, transformed_eigenvecs1, transformed_origin1)
        transformed_eigenvecs3 = self.normalize_rows(transformed_eigenvecs3)

        # Get the principal axes
        principal_axes = transformed_eigenvecs3[:, :3]  # Extract the first three eigenvectors


        # Define line segments along principal axes
        lines = []
        for i in range(3):
            endpoint = transformed_origin3 + 0.5 * principal_axes[:, i]  # Extend the line segment by a factor of 0.1
            lines.append(np.vstack((transformed_origin3, endpoint)))

        # Create line set geometry
        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(np.vstack(lines))
        line_set.lines = o3d.utility.Vector2iVector(np.array([[0, 1], [2, 3], [4, 5]]))  # Define line segments

        # Visualize the registered point clouds after ICP refinement
        o3d.visualization.draw_geometries([registered_pointcloud, camera_pointcloud])

        return transformed_eigenvecs3, transformed_origin3, fitness, registered_pointcloud, camera_pointcloud, line_set





    # def register_pcl(self, camera_pointcloud, mean, eigenvectors):
    #     OJ_EIGENVECTORS = np.array([[   -0.97345,    -0.22784,   -0.022123],
    #                                 [   -0.22528,     0.93636,     0.26923],
    #                                 [   0.040626,   -0.26707,     0.96282]])
    #     OJ_MEAN = np.array([  -0.031579,   -0.066232,    -0.44723])
    #     initial_transformation = np.eye(4)      
    #     R_new = self.compute_rotation_matrix(eigenvectors, OJ_EIGENVECTORS)
    #     initial_transformation[:3, 3] = self.compute_translation_vector(mean, OJ_MEAN, R_new)
    #     initial_transformation[:3, :3] = R_new

    #     threshold = 0.01  # 2cm distance threshold
    #     # Load your point clouds
    #     import copy
    #     ojs_pointcloud = o3d.io.read_point_cloud("/home/apo/catkin_ws/src/ur_master_thesis/src/transformed_source.ply")
    #     ojs_pointcloud_copy = copy.deepcopy(ojs_pointcloud)
    #     line_set = self.create_line_set(OJ_EIGENVECTORS, OJ_MEAN)

    #     # # Visualize the registered point clouds after ICP refinement
    #     # o3d.visualization.draw_geometries([ojs_pointcloud, line_set])

    #     # ##############################

    #     # o3d.visualization.draw_geometries([ojs_pointcloud, camera_pointcloud])
    #     ojs_pointcloud.transform(initial_transformation)
    #     transformed_eigenvecs1, transformed_origin1 = self.apply_transformation(initial_transformation, OJ_EIGENVECTORS, OJ_MEAN)
    #     transformed_eigenvecs1 = self.normalize_rows(transformed_eigenvecs1)
    #     line_set1 = self.create_line_set(transformed_eigenvecs1, transformed_origin1)
    #     # Visualize the registered point clouds after ICP refinement
    #     print("AFTER INITIAL TRANSForMATION")
    #     o3d.visualization.draw_geometries([ojs_pointcloud, line_set1, ojs_pointcloud_copy, line_set])

    #     o3d.visualization.draw_geometries([ojs_pointcloud, camera_pointcloud, line_set])
        
    #     # Downsample the point clouds
    #     voxel_size = 0.005
    #     ojs_pointcloud_downsampled = ojs_pointcloud.voxel_down_sample(voxel_size=voxel_size)
    #     camera_pointcloud_downsampled = camera_pointcloud.voxel_down_sample(voxel_size=voxel_size)

    #     # Compute normals for the downsampled point clouds
    #     ojs_pointcloud_downsampled.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    #     camera_pointcloud_downsampled.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
        
    #     # Compute FPFH features for the downsampled point clouds
    #     ojs_fpfh = o3d.pipelines.registration.compute_fpfh_feature(ojs_pointcloud_downsampled, o3d.geometry.KDTreeSearchParamHybrid(radius=0.25, max_nn=100))
    #     camera_fpfh = o3d.pipelines.registration.compute_fpfh_feature(camera_pointcloud_downsampled, o3d.geometry.KDTreeSearchParamHybrid(radius=0.25, max_nn=100))

    #     # Perform global registration using FPFH features
    #     ransac_result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
    #         ojs_pointcloud_downsampled, camera_pointcloud_downsampled,
    #         ojs_fpfh, camera_fpfh,
    #         mutual_filter=True,
    #         max_correspondence_distance=voxel_size * 2,
    #         estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
    #         ransac_n=4,
    #         checkers=[o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
    #                 o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(voxel_size * 2)],
    #         criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(4000000, 500),
    #     )

    #     # Apply the transformation from global registration
    #     global_transformation = ransac_result.transformation
    #     ojs_pointcloud.transform(global_transformation)
    #     # transformed_eigenvecs2, transformed_origin2 = self.apply_transformation(global_transformation, OJ_EIGENVECTORS, OJ_MEAN)

    #     transformed_eigenvecs2, transformed_origin2 = self.apply_transformation(global_transformation, transformed_eigenvecs1, transformed_origin1)
    #     transformed_eigenvecs2 = self.normalize_rows(transformed_eigenvecs2)
    #     line_set = self.create_line_set(transformed_eigenvecs2, transformed_origin2)
    #     # Visualize the registered point clouds after ICP refinement
    #     print("AFTER GLOBAL TRANSFORMATION")
    #     o3d.visualization.draw_geometries([ojs_pointcloud, line_set])

    #     # Visualize the registered point clouds after global registration (optional)
    #     o3d.visualization.draw_geometries([ojs_pointcloud, camera_pointcloud])

    # # Compute normals for the original point clouds (needed for ICP)
    #     ojs_pointcloud.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    #     camera_pointcloud.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    #     print("PERFORM ICP")
    #     eye = np.eye(4)
    #     # Perform ICP refinement
    #     threshold = 2 * voxel_size 
    #     icp_result = o3d.pipelines.registration.registration_icp(
    #         ojs_pointcloud, camera_pointcloud, threshold, eye,
    #         o3d.pipelines.registration.TransformationEstimationPointToPoint()
    #     )
    #     print(icp_result)
    #     print(icp_result.transformation)
    #     # Apply the transformation to the original point cloud
    #     registered_pointcloud = ojs_pointcloud.transform(icp_result.transformation)

    #     post_icp_evaluation = o3d.pipelines.registration.evaluate_registration(registered_pointcloud, camera_pointcloud, threshold, np.identity(4))
    #     print("ICP EVALUATION")
    #     print(post_icp_evaluation)

    #     # Print transformation matrix
    #     print("Global transformation matrix:")
    #     print(global_transformation)
    #     print("ICP transformation matrix:")
    #     print(icp_result.transformation)

        
        
    #     transformed_eigenvecs3, transformed_origin3 = self.apply_transformation(icp_result.transformation, transformed_eigenvecs2, transformed_origin2)
    #     transformed_eigenvecs3 = self.normalize_rows(transformed_eigenvecs3)

    #     # Get the principal axes
    #     principal_axes = transformed_eigenvecs3[:, :3]  # Extract the first three eigenvectors
    #     # principal_axes_old = eigenvectors_old[:, :3]
    #     # Define line segments along principal axes
    #     lines = []
    #     for i in range(3):
    #         endpoint = transformed_origin3 + 0.5 * principal_axes[:, i]  # Extend the line segment by a factor of 0.1
    #         lines.append(np.vstack((transformed_origin3, endpoint)))

    #     # Create line set geometry
    #     line_set = o3d.geometry.LineSet()
    #     line_set.points = o3d.utility.Vector3dVector(np.vstack(lines))
    #     line_set.lines = o3d.utility.Vector2iVector(np.array([[0, 1], [2, 3], [4, 5]]))  # Define line segments

    #     # Visualize the registered point clouds after ICP refinement
    #     o3d.visualization.draw_geometries([registered_pointcloud, camera_pointcloud, line_set])

    #     return transformed_eigenvecs3, transformed_origin3

    def estimate_grasp_point(self, rgb_image, depth_data, camera_intrinsics):
        pcl = self.__create_pcl(rgb_image, depth_data, camera_intrinsics)
        T = [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]]
        pcl = pcl.transform(T)
        pcl_surface, plane_model = self.__detect_suface(pcl)
        pcl_surface_smooth = self.__reconstruct_surface(pcl_surface)

        # Change color of smoothened surface 
        colors = np.zeros((len(pcl_surface_smooth.points), 3))
        colors[:len(pcl_surface_smooth.points)] = [255, 255, 0]  # Set existing points to green

        # Assign the colors to the point cloud
        pcl_surface_smooth.colors = o3d.utility.Vector3dVector(colors)

        # o3d.visualization.draw_geometries([pcl_surface_smooth, pcl_surface])

        # Debug step -> how good is generated pointcloud? 
        bbox, bbox_length, bbox_width = self.__get_bbox(pcl_surface_smooth)
        # bbox_new, bbox_length, bbox_width = self.__get_bbox(pcl_surface)
        # o3d.visualization.draw_geometries([pcl_surface, bbox_new, pcl])
        # Do ckeck if estimated plate makes sense -> otherwise change camera position 
        # if abs(bbox_length - self.length) > self.TOLERANCE or abs(bbox_width - self.width) > self.TOLERANCE:
        #     rospy.logwarn("ALERT: This is not a good plate estimation!")
        # o3d.visualization.draw_geometries([pcl_surface_smooth, bbox, pcl])
        # oriented_bbox = pcl_surface.get_oriented_bounding_box()
        # test_pcl = self.__get_pcl_from_bbox(oriented_bbox)


        eigenvalues, eigenvectors, mean, line_set = self.__perform_pca(pcl_surface_smooth)
        # _length = np.sqrt(eigenvalues[0])
        # _width = np.sqrt(eigenvalues[1])
        # print("Length:", _length)
        # print("Width:", _width)
        gl_1, gl_2, gw_1, gw_2, p1, p2 = self.__calculate_grasp_points(eigenvectors, mean)
        grasping_points = np.array([mean, gl_1, gl_2, gw_1, gw_2])
        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(grasping_points)


        # Create a color array for the point cloud (green for existing points, red for the added point)
        colors = np.zeros((len(grasping_points), 3))
        colors[:len(grasping_points)] = [0, 255, 0]  # Set existing points to green

        # Assign the colors to the point cloud
        point_cloud.colors = o3d.utility.Vector3dVector(colors)

        #visualize also the coordinate system
        mesh_coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin = [0,0,0])

        # Visualize the point cloud
        # o3d.visualization.draw_geometries([point_cloud, line_set, pcl, mesh_coordinate_frame])
        # o3d.io.write_point_cloud("/home/apo/catkin_ws/src/ur_master_thesis/src/transformed_source.ply", pcl)
        # print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
        # print(eigenvectors)
        # print(mean)
        transformed_eigenvecs3, transformed_origin3, fitness, registered_pointcloud, registered_pcl, registered_line_set = self.register_pcl(pcl, mean, eigenvectors)
        print("## FITNESS #########")
        print(fitness)
        if fitness > 0.9:
            print("CHOOSE REGISTRATION")
            gl_1, gl_2, gw_1, gw_2, p1, p2 = self.__calculate_grasp_points(transformed_eigenvecs3, transformed_origin3)
            grasping_points = np.array([transformed_origin3, gl_1, gl_2, gw_1, gw_2])
            point_cloud = o3d.geometry.PointCloud()
            point_cloud.points = o3d.utility.Vector3dVector(grasping_points)
            return transformed_eigenvecs3, transformed_origin3, gl_1, gl_2, gw_1, gw_2, point_cloud, registered_line_set, registered_pcl, mesh_coordinate_frame, registered_pointcloud, fitness, pcl_surface, pcl_surface_smooth
  
        rospy.logerr("Registration failed: Use PCA instead..." )
        return eigenvectors, mean, gl_1, gl_2, gw_1, gw_2, point_cloud, line_set, pcl, mesh_coordinate_frame, registered_pointcloud, fitness, pcl_surface, pcl_surface_smooth
    

if __name__ == '__main__':
    print("START")