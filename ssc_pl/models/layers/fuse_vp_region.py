import torch
import torch.nn as nn
import torch.nn.functional as F


def add_depth_embedding(tensor, positions, center):
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

class RegionAttentionModule(nn.Module):
    def __init__(self, dim, kernel_size, seq_len=208):
        super(RegionAttentionModule, self).__init__()
        self.dim = dim
        self.kernel_size = kernel_size
        self.seq_len = seq_len
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channels=128, out_channels=dim, kernel_size=kernel_size, stride=2, padding=kernel_size // 2)
        self.bn1 = nn.BatchNorm2d(dim)
        self.conv2 = nn.Conv2d(in_channels=dim, out_channels=dim * 2, kernel_size=kernel_size, stride=2, padding=kernel_size // 2)
        self.bn2 = nn.BatchNorm2d(dim * 2)
        self.conv3 = nn.Conv2d(in_channels=dim * 2, out_channels=dim * 4, kernel_size=kernel_size, stride=1, padding=kernel_size // 2)
        self.bn3 = nn.BatchNorm2d(dim * 4)
        
        # Bottleneck 1x1 convolution
        self.bottleneck = nn.Conv2d(in_channels=dim * 4, out_channels=dim, kernel_size=1)
        self.bottleneck_bn = nn.BatchNorm2d(dim)
        
        # Learnable positional embedding
        self.pos_embedding = nn.Embedding(self.seq_len, dim)  # Positional embeddings for seq_len positions

        # Advanced attention mechanism
        self.attention = nn.MultiheadAttention(embed_dim=dim, num_heads=16, batch_first=True)
        
        # Optional dropout for regularization
        self.dropout = nn.Dropout(p=0.1)
    
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
            
            # Dynamically calculate padding
            pad_x = max(0, width - crop.size(2))
            pad_y = max(0, height - crop.size(1))
            padded_crop = F.pad(crop, (0, pad_x, 0, pad_y))
            crops.append(padded_crop)
        
        return torch.stack(crops)

    def forward(self, img_feat, center, norm_points, queries):
        _, n, _ = norm_points.shape
        norm_points = norm_points[0]
        bs, c, h, w = img_feat.shape
        width, height = w // 3, h // 3

        # Crop and pad image features
        cropped_padded_features = self.crop_and_pad(img_feat, center, width, height)
        # print(cropped_padded_features.shape)

        # print(img_feat.shape)

        # Apply CNN to extract 1D features
        x = F.relu(self.bn1(self.conv1(cropped_padded_features)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))

        # Apply bottleneck
        x = F.relu(self.bottleneck_bn(self.bottleneck(x))).reshape(bs, 128, -1)
        x = torch.permute(x, (0, 2, 1))  # Shape: (bs, seq_len, dim)

        # Add positional embeddings to CNN features
        pos_ids = torch.arange(self.seq_len, device=img_feat.device).unsqueeze(0)  # Shape: (1, seq_len)
        x = x + self.pos_embedding(pos_ids)

        # Cross-attention
        queries_with_pos = queries  # Assuming queries already have embeddings or additional embeddings can be added here
        query = self.dropout(queries_with_pos)
        key = self.dropout(x.view(bs, -1, self.dim))
        value = key

        attn_output, _ = self.attention(query, key, value)
        return attn_output.view(bs, n, -1)

