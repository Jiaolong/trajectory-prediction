import math
import random
import numpy as np
from pathlib import Path
from sklearn.cluster import KMeans

import torch
from torch import nn
from torch.nn import functional as F

class MTP(nn.Module):
    """
    Implementation of Multiple-Trajectory Prediction (MTP) model
    based on https://arxiv.org/pdf/1809.10732.pdf
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.num_modes = cfg.num_modes
        self.input_dim = cfg.input_dim
        predictions_per_mode = cfg.num_points_per_trajectory * 2

        if cfg.get('with_hist_traj'):
            self.input_dim += predictions_per_mode

        self.fc1 = nn.Linear(self.input_dim, cfg.n_hidden_layers)

        self.fc2 = nn.Linear(cfg.n_hidden_layers, int(self.num_modes * predictions_per_mode + self.num_modes))

        self.loss_function = MTPLoss(self.num_modes, 1, 5)
       
        if cfg.get('with_anchor'):
            traj_set_path = Path(cfg.anchor_path) / 'trajset.npy'
            if traj_set_path.exists():
                traj_set = np.load(traj_set_path)
                X = traj_set.reshape((traj_set.shape[0], -1))
                kmeans = KMeans(n_clusters=cfg.num_modes, random_state=0).fit(X)
                anchors = kmeans.cluster_centers_
                print('Obtained {} anchor trajectories from {}'.format(anchors.shape[0], traj_set_path))
                anchors = anchors.reshape((cfg.num_modes, traj_set.shape[1], traj_set.shape[2]))
                if cfg.get('normalize_trajectory', False):
                    anchors = anchors / cfg.norm_factor
                self.anchors = torch.from_numpy(anchors).float().cuda()
                self.anchors = self.anchors.unsqueeze(0)
            else:
                raise ValueError("Trajectory set is not found at {}".format(traj_set_path))

    def forward(self, batch_dict):
        """
        Forward pass of the model.
        Returns:  
            - trajectory_predictions: [batch_size, number_of_modes * number_of_predictions_per_mode]. 
            Storing the predicted trajectory and mode probabilities. 
            - mode_probabilities: [batch_size, number_of_modes]. 
            Mode probabilities are normalized to sum to 1 during inference.
        """
        features = batch_dict['reg_features']
        if self.cfg.get('with_hist_traj'):
            traj_hist = batch_dict['traj_hist'].view(features.shape[0], -1)
            features = torch.cat([features, traj_hist], dim=1)

        predictions = self.fc2(self.fc1(features))

        # Normalize the probabilities to sum to 1 for inference.
        mode_probabilities = predictions[:, -self.num_modes:]
        if not self.training:
            mode_probabilities = F.softmax(mode_probabilities, dim=-1)

        predictions = predictions[:, :-self.num_modes]
        
        # reshape [batch_size, n_modes, n_points, 2]
        traj_shape = (features.shape[0], self.num_modes, -1, 2)
        predictions = predictions.reshape(traj_shape)
        if self.cfg.get('with_anchor'):
            predictions += self.anchors

        batch_dict['trajectory_predictions'] = predictions
        batch_dict['mode_probabilities'] = mode_probabilities

        if self.training:
            self.trajectory_predictions = predictions
            self.mode_probabilities = mode_probabilities
            self.ground_truth = batch_dict['traj_ins']
        return batch_dict

    def get_loss(self):
        loss = self.loss_function(self.trajectory_predictions, self.mode_probabilities, self.ground_truth)
        return loss

class MTPLoss:
    """ Computes the loss for the MTP model. """

    def __init__(self,
                 num_modes: int,
                 regression_loss_weight: float = 1.,
                 angle_threshold_degrees: float = 5.):
        """
        Inits MTP loss.
        :param num_modes: How many modes are being predicted for each agent.
        :param regression_loss_weight: Coefficient applied to the regression loss to
            balance classification and regression performance.
        :param angle_threshold_degrees: Minimum angle needed between a predicted trajectory
            and the ground to consider it a match.
        """
        self.num_modes = num_modes
        self.regression_loss_weight = regression_loss_weight
        self.angle_threshold = angle_threshold_degrees

    @staticmethod
    def _angle_between(ref_traj, traj_to_compare):
        """
        Computes the angle between the last points of the two trajectories.
        The resulting angle is in degrees and is an angle in the [0; 180) interval.
        :param ref_traj: Tensor of shape [n_timesteps, 2].
        :param traj_to_compare: Tensor of shape [n_timesteps, 2].
        :return: Angle between the trajectories.
        """
        EPSILON = 1e-5
        if (ref_traj.ndim != 2 or traj_to_compare.ndim != 2 or
                ref_traj.shape[1] != 2 or traj_to_compare.shape[1] != 2):
            raise ValueError('Both tensors should have shapes (-1, 2).')

        if torch.isnan(traj_to_compare[-1]).any() or torch.isnan(ref_traj[-1]).any():
            return 180. - EPSILON

        traj_norms_product = float(torch.norm(ref_traj[-1]) * torch.norm(traj_to_compare[-1]))

        # If either of the vectors described in the docstring has norm 0, return 0 as the angle.
        if math.isclose(traj_norms_product, 0):
            return 0.

        # We apply the max and min operations below to ensure there is no value
        # returned for cos_angle that is greater than 1 or less than -1.
        # This should never be the case, but the check is in place for cases where
        # we might encounter numerical instability.
        dot_product = float(ref_traj[-1].dot(traj_to_compare[-1]))
        angle = math.degrees(math.acos(max(min(dot_product / traj_norms_product, 1), -1)))

        if angle >= 180:
            return angle - EPSILON

        return angle

    @staticmethod
    def _compute_ave_l2_norms(tensor):
        """
        Compute the average of l2 norms of each row in the tensor.
        :param tensor: Shape [1, n_timesteps, 2].
        :return: Average l2 norm. Float.
        """
        l2_norms = torch.norm(tensor, p=2, dim=2)
        avg_distance = torch.mean(l2_norms)
        return avg_distance.item()

    def _compute_angles_from_ground_truth(self, target, trajectories):
        """
        Compute angle between the target trajectory (ground truth) and the predicted trajectories.
        :param target: Shape [n_points, 2].
        :param trajectories: Shape [n_modes, n_points, 2].
        :return: List of angle, index tuples.
        """
        angles_from_ground_truth = []
        for mode, mode_trajectory in enumerate(trajectories):
            # For each mode, we compute the angle between the last point of the predicted trajectory for that
            # mode and the last point of the ground truth trajectory.
            angle = self._angle_between(target, mode_trajectory)

            angles_from_ground_truth.append((angle, mode))
        return angles_from_ground_truth

    def _compute_best_mode(self, angles_from_ground_truth, target, trajectories):
        """
        Finds the index of the best mode given the angles from the ground truth.
        :param angles_from_ground_truth: List of (angle, mode index) tuples.
        :param target: Shape [n_points, 2]
        :param trajectories: Shape [n_modes, n_points, 2]
        :return: Integer index of best mode.
        """

        # We first sort the modes based on the angle to the ground truth (ascending order), and keep track of
        # the index corresponding to the biggest angle that is still smaller than a threshold value.
        angles_from_ground_truth = sorted(angles_from_ground_truth)
        max_angle_below_thresh_idx = -1
        for angle_idx, (angle, mode) in enumerate(angles_from_ground_truth):
            if angle <= self.angle_threshold:
                max_angle_below_thresh_idx = angle_idx
            else:
                break

        # We choose the best mode at random IF there are no modes with an angle less than the threshold.
        if max_angle_below_thresh_idx == -1:
            best_mode = random.randint(0, self.num_modes - 1)

        # We choose the best mode to be the one that provides the lowest ave of l2 norms between the
        # predicted trajectory and the ground truth, taking into account only the modes with an angle
        # less than the threshold IF there is at least one mode with an angle less than the threshold.
        else:
            # Out of the selected modes above, we choose the final best mode as that which returns the
            # smallest ave of l2 norms between the predicted and ground truth trajectories.
            distances_from_ground_truth = []

            for angle, mode in angles_from_ground_truth[:max_angle_below_thresh_idx + 1]:
                norm = self._compute_ave_l2_norms(target.unsqueeze(0) - trajectories[mode, :, :])

                distances_from_ground_truth.append((norm, mode))

            distances_from_ground_truth = sorted(distances_from_ground_truth)
            best_mode = distances_from_ground_truth[0][1]

        return best_mode

    def __call__(self, trajectories, modes, targets):
        """
        Computes the MTP loss on a batch.
        Inputs:
         - trajectories: Trajectory predictions for batch.
         - modes: Mode predictions for batch.
         - targets: Targets for batch. Targets are of shape [batch_size, 1, n_points, 2]
        Return: zero-dim tensor representing the loss on the batch.
        """
        batch_losses = torch.Tensor().requires_grad_(True).to(trajectories.device)

        for batch_idx in range(trajectories.shape[0]):

            angles = self._compute_angles_from_ground_truth(target=targets[batch_idx],
                                                            trajectories=trajectories[batch_idx])

            best_mode = self._compute_best_mode(angles,
                                                target=targets[batch_idx],
                                                trajectories=trajectories[batch_idx])

            best_mode_trajectory = trajectories[batch_idx, best_mode, :]
            regression_loss = F.smooth_l1_loss(best_mode_trajectory, targets[batch_idx])

            mode_probabilities = modes[batch_idx].unsqueeze(0)
            best_mode_target = torch.tensor([best_mode], device=trajectories.device)
            classification_loss = F.cross_entropy(mode_probabilities, best_mode_target)

            loss = classification_loss + self.regression_loss_weight * regression_loss

            batch_losses = torch.cat((batch_losses, loss.unsqueeze(0)), 0)

        avg_loss = torch.mean(batch_losses)

        return avg_loss
