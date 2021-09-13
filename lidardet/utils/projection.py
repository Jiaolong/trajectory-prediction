import numpy as np

class PointProjection(object):

    def __init__(self, cfg):
        self.cfg = cfg

    @staticmethod
    def xyz_to_cylinder(points_xyz):
        points_x, points_y, points_z = np.split(points_xyz, 3, axis=-1)
        points_rho = np.sqrt(points_x**2 + points_y**2)
        points_phi = np.arctan2(points_y, points_x)
        points_cylinder = np.stack([points_phi, points_z, points_rho], axis=-1)
        return points_cylinder

    @staticmethod
    def cylinder_to_xyz(points_cylinder):
        points_phi, points_z, points_rho = np.split(points_cylinder, 3, axis=-1)
        points_x = points_rho * np.cos(points_phi)
        points_y = points_rho * np.sin(points_phi)
        points_xyz = np.stack([points_x, points_y, points_z], axis=-1)
        return points_xyz

    def get_kitti_columns(points: np.array, number_of_columns: int = 2000) -> np.array:
        """ Returns the column indices for unfolding one or more raw KITTI scans """
        azi = np.arctan2(points[..., 1], points[..., 0])
        cols = number_of_columns * (np.pi - azi) / (2 * np.pi)
        # In rare cases it happens that the index is exactly one too large.
        cols = np.minimum(cols, number_of_columns - 1)
        return np.int32(cols)


    def get_kitti_rows(points: np.array, threshold: float = -0.005) -> np.array:
        """ Returns the row indices for unfolding one or more raw KITTI scans """
        azimuth_flipped = -np.arctan2(points[..., 1], -points[..., 0])
        azi_diffs = azimuth_flipped[..., 1:] - azimuth_flipped[..., :-1]
        jump_mask = np.greater(threshold, azi_diffs)
        ind = np.add(np.where(jump_mask), 1)
        rows = np.zeros_like(points[..., 0])
        rows[..., ind] += 1
        return np.int32(np.cumsum(rows, axis=-1))


    def projection(points: np.array, *channels, image_size: tuple = (64, 2000)):
        """ 
        Scan unfolding of raw KITTI point cloud of shape [N, (x,y,z)]
        This functions performs a cylindrical projection of a 3D point cloud given in
        cartesian coordinates. The function aims to recover the internal data structure
        of the KITTI HDL-64 sensor from its distinct scan pattern that can be approximated
        from the original KITTI Raw Dataset.
        The resulting projected image (H, W) contains missing values, these positions
        are filled with `-1`. If you set `return_inverse` to True, you may receive
        a mask for the valid measurements of the image with `valid_image = inverse >= 0`.
        This function has less information loss than techniques which compute the
        azimuth and elevation angle and sort the points accordingly into the image.
        Therefore, most entries in `valid_image` shall be True, except for locations
        that hit the ego vehicle or are pointed towards the sky.
        The measurements are ordered in decreasing depth before being assigned to image
        locations. In case more than one value is scattered to this position, the one
        closest to the ego vehicle is selected. To know which points actually made it into
        the projection, set `return_active` to True.
        Arguments:
            points : np.array(shape=(N, 3), dtype=np.float)
                 The raw KITTI point cloud (does not work with ego-motion corrected data!).
            image_size : tuple of 2 integers
                The image size of the projection in (H, W). (Default is (64, 2000).
            channels : * times np.array(shape=(N, D))
                Additional channels to project in the same way as `points`.
        Returns:
            Dict(
                points=np.array(shape=(H, W, 3), dtype=np.float),  # projected point cloud
                depth=np.array(shape=(N,), dtype=np.float)  # projected depth
                channels=List(*),  # list of projected additional channels
                indices=np.array(shape=(N, 2), dtype=np.int)  # image indices from the list
                inverse=np.array(shape=(H, W), dtype=np.int)  # list indices from the image
                active=np.array(shape=(N,), dtype=np.bool)  # active array
        Raises:
            IndexError if projection is not possible, e.g. if you do not use a KITTI
                point cloud, or the KITTI point cloud is ego motion corrected
        """

        output = {}

        assert points.shape[1] == 3, "Points must contain N xyz coordinates."
        if len(channels) > 0:
            assert all(
                isinstance(x, np.ndarray) for x in channels
            ), "All channels must be numpy arrays."
            assert all(
                x.shape[0] == points.shape[0] for x in channels
            ), "All channels must have the same first dimension as `points`."

        # Get depth of all points for ordering.
        depth_list = np.linalg.norm(points, 2, axis=1)

        # Get the indices of the rows and columns to project to.
        proj_column = get_kitti_columns(points, number_of_columns=image_size[1])
        proj_row = get_kitti_rows(points)

        if np.any(proj_row >= image_size[0]) or np.any(proj_column >= image_size[1]):
            raise IndexError(
                "Cannot find valid image indices for this point cloud and image size. "
                "Are you sure you entered the correct image size? This function only works "
                "with raw KITTI HDL-64 point clouds (no ego motion corrected data allowed)!"
            )

        # Store a copy in original order.
        output["indices"] = np.stack([np.copy(proj_row), np.copy(proj_column)], axis=-1)

        # Get the indices in order of decreasing depth.
        indices = np.arange(depth_list.shape[0])
        order = np.argsort(depth_list)[::-1]

        indices = indices[order]
        proj_column = proj_column[order]
        proj_row = proj_row[order]

        # Project the points.
        points_img = np.full(shape=(*image_size, 3), fill_value=-1, dtype=np.float32)
        points_img[proj_row, proj_column] = points[order]
        output["points"] = points_img

        # The depth projection.
        depth_img = np.full(shape=image_size, fill_value=-1, dtype=np.float32)
        depth_img[proj_row, proj_column] = depth_list[order]
        output["depth"] = depth_img

        # Convert all channels.
        projected_channels = []
        for channel in channels:
            # Set the shape.
            _shape = (
                (*image_size, channel.shape[1])
                if len(channel.shape) > 1
                else (*image_size,)
            )

            # Initialize the image.
            _image = np.full(shape=_shape, fill_value=-1, dtype=channel.dtype)

            # Assign the values.
            _image[proj_row, proj_column] = channel[order]
            projected_channels.append(_image)
        output["channels"] = projected_channels

        # Get the inverse indices mapping.
        list_indices_img = np.full(image_size, -1, dtype=np.int32)
        list_indices_img[proj_row, proj_column] = indices
        output["inverse"] = list_indices_img

        # Set which points are used in the projection.
        active_list = np.full(depth_list.shape, fill_value=0, dtype=np.int32)
        active_list[list_indices_img] = 1
        output["active"] = active_list.astype(np.bool)

        return output
