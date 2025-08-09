import torch
import torch.nn as nn
import torch.nn.functional as F
import time

def sample_patches(coord_x, coord_y, feat, C, H, W, half_k, kernel_size, n, bs):
    # Initialize patches with zeros
    patches = torch.zeros((bs, C, n, kernel_size * kernel_size), device=feat.device, dtype=feat.dtype)

    idx = 0
    for j in range(-half_k, half_k + 1):
        for k in range(-half_k, half_k + 1):
            # Calculate the offset coordinates
            offset_x = torch.clamp(coord_x + k, 0, W - 1)
            offset_y = torch.clamp(coord_y + j, 0, H - 1)

            # Sample the feature value at the given coordinate
            for c in range(C):
                patches[:, c, :, idx] = feat[:, c, offset_y.long(), offset_x.long()]

            idx += 1
    return patches

def sample_patches_parallel(coord_x, coord_y, feat, H, W, offsets):
    # Calculate offset coordinates
    offset_x = coord_x.unsqueeze(-1) + offsets[:, 1].view(1, 1, -1)
    offset_y = coord_y.unsqueeze(-1) + offsets[:, 0].view(1, 1, -1)

    # Clamp the coordinates to stay within valid bounds
    offset_x = torch.clamp(offset_x, 0, W - 1).long()
    offset_y = torch.clamp(offset_y, 0, H - 1).long()

    # Sample the feature values at the offset coordinates for all channels in parallel
    patches = feat[:, :, offset_y, offset_x]  # Shape: (bs, C, n, kernel_size*kernel_size)

    return patches

class MultiLevelFeatureSampler(nn.Module):
    def __init__(self, output_dim, kernel_sizes=[7, 5, 3]):
        super(MultiLevelFeatureSampler, self).__init__()
        self.kernel_sizes = kernel_sizes
        # Calculate the total input dimension for the linear layer
        self.input_dim = sum([k * k for k in kernel_sizes])
        self.output_dim = output_dim
        # Define the linear layer for projecting to the desired dimension
        self.fc = nn.Linear(self.input_dim, output_dim)

    def forward(self, points, features):
        """
        Sample multi-level features and project them using a feedforward layer.
        
        Args:
        - points (torch.Tensor): Normalized 2D positions (bs, n, 2).
        - features (list[torch.Tensor]): List of multi-level feature maps with shapes
                                         [(bs, C, H1, W1), (bs, C, H2, W2), (bs, C, H3, W3)].
        
        Returns:
        - torch.Tensor: Output tensor of shape (bs, C, n, output_dim).
        """
        device = features[0].device
        bs = features[0].shape[0]
        C = features[0].shape[1]
        n = points.shape[1]

        sampled_patches = []

        for i, feat in enumerate(features):
            H, W = feat.shape[2], feat.shape[3]

            # Rescale normalized coordinates to the feature map size
            x = (points[:, :, 0] * (W - 1)).clamp(0, W - 1)
            y = (points[:, :, 1] * (H - 1)).clamp(0, H - 1)

            kernel_size = self.kernel_sizes[i]
            half_k = kernel_size // 2

            offsets = torch.tensor([(j, k) for j in range(-half_k, half_k + 1) for k in range(-half_k, half_k + 1)],
                device=feat.device, dtype=feat.dtype)

            patches = sample_patches_parallel(x, y, feat, H, W, offsets).squeeze(2)
            sampled_patches.append(patches)  # Shape: (bs, C, n, kernel_size*kernel_size)

        # Concatenate all levels along the last dimension
        concatenated = torch.cat(sampled_patches, dim=-1)  # Shape: (bs, C, n, sum(kernel_sizes[i]*kernel_sizes[i]))

        # Reshape for the feedforward layer: (bs, C, n, input_dim) -> (bs*C*n, input_dim)
        concatenated = concatenated.permute(0, 2, 1, 3).reshape(bs * C * n, self.input_dim)

        # Apply the feedforward layer
        projected = self.fc(concatenated)  # Shape: (bs*C*n, output_dim)

        # Reshape back to (bs, C, n, output_dim)
        projected = projected.view(bs, C, n, self.output_dim)
        return projected
