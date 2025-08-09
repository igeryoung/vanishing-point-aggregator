import torch
import torch.nn as nn
import torch.nn.functional as F


def add_position_embedding(tensor, positions, center):
    """
    Add position embeddings to a tensor based on L1 distance from a given center.

    Parameters:
    - tensor: A PyTorch tensor of shape (bs, n, dim).
    - positions: A tensor of shape (n, 2) representing the 2D positions of each element.
    - center: A tuple (x, y) representing the center position.

    Returns:
    - A new tensor with position embeddings added to the original tensor.
    """
    bs, n, embed_dim = tensor.shape

    # Calculate the L1 distance (Manhattan distance) from each position to the center
    l1_distances = torch.abs(positions - torch.tensor(center)).sum(dim=1)  # Shape: (n,)

    # Normalize the distances to the range [0, 1]
    l1_distances = (l1_distances - l1_distances.min()) / (l1_distances.max() - l1_distances.min() + 1e-6)

    # Create position embeddings based on the L1 distances
    position_embeddings = l1_distances.unsqueeze(1).repeat(1, embed_dim)  # Shape: (n, embed_dim)

    # Expand position embeddings to match the batch size
    position_embeddings = position_embeddings.unsqueeze(0).expand(bs, -1, -1)  # Shape: (bs, n, embed_dim)

    # Concatenate the position embeddings with the original tensor
    tensor_with_position = tensor + position_embeddings
    # tensor_with_position = torch.cat([, position_embeddings], dim=-1)  # Shape: (bs, n, dim + embed_dim)

    return tensor_with_position

class CrossAttentionModule(nn.Module):
    def __init__(self, dim, kernel_size):
        super(CrossAttentionModule, self).__init__()
        self.dim = dim
        self.kernel_size = kernel_size
        self.conv = nn.Conv2d(in_channels=128, out_channels=dim, kernel_size=kernel_size, stride=3, padding=0)
        self.attention = nn.MultiheadAttention(embed_dim=dim, num_heads=8, batch_first=True)
    
    def crop_and_pad(self, img_feat, centers, width, height):
        bs, c, h, w = img_feat.shape
        crops = []
        cx, cy = centers
        for i in range(bs):
            cx, cy = int(cx * w), int(cy * h)
            x1 = max(0, cx - width // 2)
            x2 = min(w, cx + width // 2)
            y1 = max(0, cy - height // 2)
            y2 = min(h, cy + height // 2)
            
            crop = img_feat[i, :, y1:y2, x1:x2]
            pad_x = 102 - crop.size(2)
            pad_y = 31 - crop.size(1)
            padded_crop = F.pad(crop, (0, pad_x, 0, pad_y))
            crops.append(padded_crop)
        
        return torch.stack(crops)
    
    def forward(self, img_feat, center, norm_points, instance_queries):
        # Constants
        n, _ = norm_points.shape
        bs, c, h, w = img_feat.shape
        width = w // 3
        height = h // 3

        # Crop and pad image features
        cropped_padded_features = self.crop_and_pad(img_feat, center, width, height)
        
        # Apply CNN to extract 1D features from the cropped features
        sampled_kernels = self.conv(cropped_padded_features).view(bs, -1, self.dim)
        
        denormalize_points = norm_points* torch.tensor([1220, 370])
        denormalize_center = center* torch.tensor([1220, 370])
        instance_queries_with_pos = add_position_embedding(instance_queries, denormalize_points, denormalize_center)
        
        # Cross-attention
        query = instance_queries_with_pos
        key = sampled_kernels.view(bs, -1, self.dim)
        value = key

        print(query.shape)
        print(key.shape)
        
        attn_output, _ = self.attention(query, key, value)
        return attn_output.view(bs, n, -1)

# Example usage
bs, n, dim = 2, 5, 64
image_features = torch.randn(bs, 128, 93, 305)  # Example image feature
norm_points = torch.rand(n, 2)  # Normalized points between [0, 1]
instance_queries = torch.randn(bs, n, dim)  # Instance queries
center = torch.tensor([0.6, 0.5])

cross_attn_module = CrossAttentionModule(dim=dim, kernel_size=5)
output = cross_attn_module(image_features, center, norm_points, instance_queries)
print(output.shape)
