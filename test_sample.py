import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from numba import njit, prange

@njit(parallel=True)
def sample_patches(coord_x, coord_y, feat_numpy, C, H, W, half_k, kernel_size, n, bs):
    patches = np.zeros((bs, C, n, kernel_size * kernel_size), dtype=np.float32)

    idx = 0
    for j in range(-half_k, half_k + 1):
        for k in range(-half_k, half_k + 1):
            for b in prange(bs):
                for i in prange(n):
                    # Offset coordinates
                    offset_x = min(max(coord_x[b, i] + k, 0), W - 1)
                    offset_y = min(max(coord_y[b, i] + j, 0), H - 1)

                    # Sample the feature value at the given coordinate
                    for c in range(C):
                        patches[b, c, i, idx] = feat_numpy[b, c, int(offset_y), int(offset_x)]
            idx += 1
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

        # Convert normalized points to feature map coordinates
        coords = []
        for i, feat in enumerate(features):
            H, W = feat.shape[2], feat.shape[3]
            # Rescale normalized coordinates to the feature map size
            x = (points[:, :, 0] * (W - 1)).clamp(0, W - 1).cpu().numpy()
            y = (points[:, :, 1] * (H - 1)).clamp(0, H - 1).cpu().numpy()
            coords.append((x, y))

        sampled_patches = []

        for i, (feat, (coord_x, coord_y), kernel_size) in enumerate(zip(features, coords, self.kernel_sizes)):
            half_k = kernel_size // 2
            H, W = feat.shape[2], feat.shape[3]

            # Convert the feature map to a NumPy array
            feat_numpy = feat.cpu().numpy()

            # Use Numba to sample patches
            patches_numpy = sample_patches(coord_x, coord_y, feat_numpy, C, H, W, half_k, kernel_size, n, bs)

            # Convert back to PyTorch tensor and move to the original device
            patches = torch.from_numpy(patches_numpy).to(device)  # Shape: (bs, C, n, kernel_size*kernel_size)

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

# Example usage
bs = 4
points = torch.rand(bs, 10, 2).to('cuda')  # bs = 4, n = 10, 2D normalized points, move to GPU if available
features = [
    torch.randn(bs, 128, 93, 305).to('cuda'),  # Level 0 feature map
    torch.randn(bs, 128, 47, 153).to('cuda'),  # Level 1 feature map
    torch.randn(bs, 128, 24, 77).to('cuda')    # Level 2 feature map
]

output_dim = 256
sampler = MultiLevelFeatureSampler(output_dim).to('cuda')
output = sampler(points, features)
print(output.shape)  # Expected shape: (4, 128, 10, 256)
