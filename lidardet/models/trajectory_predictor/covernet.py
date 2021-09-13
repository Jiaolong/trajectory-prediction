import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

class CoverNet(nn.Module):
    """ Implementation of CoverNet https://arxiv.org/pdf/1911.10298.pdf """

    def __init__(self, cfg):
        """
        Inits Covernet.
        """

        super().__init__()

        n_hidden_layers = cfg.get('n_hidden_layers', [4096])
        num_modes = cfg.num_modes
        n_hidden_layers = [cfg.input_dim] + n_hidden_layers + [num_modes]

        linear_layers = [nn.Linear(in_dim, out_dim)
                         for in_dim, out_dim in zip(n_hidden_layers[:-1], n_hidden_layers[1:])]

        self.head = nn.ModuleList(linear_layers)
        
        self.loss_function = nn.CrossEntropyLoss()

    def forward(self, batch_dict): 

        logits = batch_dict['reg_features']
        for linear in self.head:
            logits = linear(logits)


        if self.training:
            self.logits = logits
            self.label = batch_dict['traj_label']
        else:
            logits = F.softmax(logits, dim=-1)
        
        batch_dict['logits'] = logits
        return batch_dict

    def get_loss(self):
        loss = self.loss_function(self.logits, self.label.long())
        return loss

def mean_pointwise_l2_distance(lattice, ground_truth):
    """
    Computes the index of the closest trajectory in the lattice as measured by l1 distance.
    :param lattice: Lattice of pre-generated trajectories. Shape [num_modes, n_timesteps, state_dim]
    :param ground_truth: Ground truth trajectory of agent. Shape [1, n_timesteps, state_dim].
    :return: Index of closest mode in the lattice.
    """
    stacked_ground_truth = ground_truth.repeat(lattice.shape[0], 1, 1)
    return torch.pow(lattice - stacked_ground_truth, 2).sum(dim=2).sqrt().mean(dim=1).argmin()


class ConstantLatticeLoss:
    """
    Computes the loss for a constant lattice CoverNet model.
    """

    def __init__(self, lattice, similarity_function = mean_pointwise_l2_distance):
        """
        Inits the loss.
        :param lattice: numpy array of shape [n_modes, n_timesteps, state_dim]
        :param similarity_function: Function that computes the index of the closest trajectory in the lattice
            to the actual ground truth trajectory of the agent.
        """

        self.lattice = torch.Tensor(lattice)
        self.similarity_func = similarity_function

    def __call__(self, batch_logits, batch_ground_truth_trajectory):
        """
        Computes the loss on a batch.
        :param batch_logits: Tensor of shape [batch_size, n_modes]. Output of a linear layer since this class
            uses nn.functional.cross_entropy.
        :param batch_ground_truth_trajectory: Tensor of shape [batch_size, n_timesteps, state_dim]
        :return: Average element-wise loss on the batch.
        """

        # If using GPU, need to copy the lattice to the GPU if haven't done so already
        # This ensures we only copy it once
        if self.lattice.device != batch_logits.device:
            self.lattice = self.lattice.to(batch_logits.device)

        batch_losses = torch.Tensor().requires_grad_(True).to(batch_logits.device)

        for logit, ground_truth in zip(batch_logits, batch_ground_truth_trajectory):

            closest_lattice_trajectory = self.similarity_func(self.lattice, ground_truth.unsqueeze(0))
            label = torch.LongTensor([closest_lattice_trajectory]).to(batch_logits.device)
            classification_loss = F.cross_entropy(logit, label)

            batch_losses = torch.cat((batch_losses, classification_loss.unsqueeze(0)), 0)

        return batch_losses.mean()
