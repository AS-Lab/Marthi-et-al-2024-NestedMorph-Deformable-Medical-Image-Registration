# Import necessary libraries
import torch
import torch.nn as nn
from einops import rearrange
from einops.layers.torch import Rearrange
from torch.nn import functional as F
from PIL import Image
import numpy as np
import os
from torch.distributions.normal import Normal

def drop_path(x, drop_prob: float = 0., training: bool = False, scale_by_keep: bool = True):
    """
    Applies stochastic depth (drop paths) to the input tensor during training.

    Args:
        x (Tensor): Input tensor.
        drop_prob (float): Probability of dropping a path (default: 0).
        training (bool): Whether the model is in training mode (default: False).
        scale_by_keep (bool): Whether to scale the output by the keep probability (default: True).

    Returns:
        Tensor: The input tensor with dropped paths.
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor


class DropPath(nn.Module):
    """
    Drop paths (Stochastic Depth) layer for use in residual blocks.
    """
    def __init__(self, drop_prob: float = 0., scale_by_keep: bool = True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        """Forward pass."""
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)

    def extra_repr(self):
        return f'drop_prob={round(self.drop_prob,3):0.3f}'

class DWConv(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dwconv = nn.Conv3d(dim, dim, 3, 1, 1, groups=dim)

    def forward(self, x: torch.Tensor, D, H, W) -> torch.Tensor:
        B, N, C = x.shape
        tx = x.transpose(1, 2).view(B, C, D, H, W)
        conv_x = self.dwconv(tx)
        return conv_x.flatten(2).transpose(1, 2)

class MixFFN(nn.Module):
    def __init__(self, c1, c2):
        super().__init__()
        self.fc1 = nn.Linear(c1, c2)
        self.dwconv = DWConv(c2)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(c2, c1)

    def forward(self, x: torch.Tensor, D, H, W) -> torch.Tensor:
        ax = self.act(self.dwconv(self.fc1(x), D, H, W))
        out = self.fc2(ax)
        return out

class MixFFN_skip(nn.Module):
    def __init__(self, c1, c2):
        super().__init__()
        self.fc1 = nn.Linear(c1, c2)
        self.dwconv = DWConv(c2)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(c2, c1)
        self.norm1 = nn.LayerNorm(c2)
        self.norm2 = nn.LayerNorm(c2)
        self.norm3 = nn.LayerNorm(c2)

    def forward(self, x: torch.Tensor, D, H, W) -> torch.Tensor:
        ax = self.act(self.norm1(self.dwconv(self.fc1(x), D, H, W) + self.fc1(x)))
        out = self.fc2(ax)
        return out

class DWConvLKA(nn.Module):
    """
    Depthwise convolution with group-wise convolution applied to the input tensor.
    """
    def __init__(self, dim=768):
        super(DWConvLKA, self).__init__()
        self.dwconv = nn.Conv3d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x):
        """Forward pass."""
        x = self.dwconv(x)
        return x

class GlobalExtraction(nn.Module):
    def __init__(self, dim=None):
        super().__init__()
        self.avgpool = self.globalavgchannelpool
        self.maxpool = self.globalmaxchannelpool
        self.proj = nn.Sequential(
            nn.Conv3d(2, 1, 1, 1),
            nn.BatchNorm3d(1)
        )

    def globalavgchannelpool(self, x):
        x = x.mean(1, keepdim=True)
        return x

    def globalmaxchannelpool(self, x):
        x = x.max(dim=1, keepdim=True)[0]
        return x

    def forward(self, x):
        x_ = x.clone()
        x = self.avgpool(x)
        x2 = self.maxpool(x_)

        cat = torch.cat((x, x2), dim=1)

        proj = self.proj(cat)
        return proj

class ContextExtraction(nn.Module):
    def __init__(self, dim, reduction=None):
        super().__init__()
        self.reduction = 1 if reduction is None else 2

        self.dconv = self.DepthWiseConv3dx2(dim)
        self.proj = self.Proj(dim)

    def DepthWiseConv3dx2(self, dim):
        dconv = nn.Sequential(
            nn.Conv3d(in_channels=dim,
                      out_channels=dim,
                      kernel_size=3,
                      padding=1,
                      groups=dim),
            nn.BatchNorm3d(num_features=dim),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels=dim,
                      out_channels=dim,
                      kernel_size=3,
                      padding=2,
                      dilation=2),
            nn.BatchNorm3d(num_features=dim),
            nn.ReLU(inplace=True)
        )
        return dconv

    def Proj(self, dim):
        proj = nn.Sequential(
            nn.Conv3d(in_channels=dim,
                      out_channels=dim // self.reduction,
                      kernel_size=1),
            nn.BatchNorm3d(num_features=dim // self.reduction)
        )
        return proj

    def forward(self, x):
        x = self.dconv(x)
        x = self.proj(x)
        return x

class MultiscaleFusion(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.local = ContextExtraction(dim)
        self.global_ = GlobalExtraction()
        self.bn = nn.BatchNorm3d(num_features=dim)

    def forward(self, x, g):
        x = self.local(x)
        g = self.global_(g)

        fuse = self.bn(x + g)
        return fuse


class MultiScaleGatedAttn(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.multi = MultiscaleFusion(dim)
        self.selection = nn.Conv3d(dim, 2, 1)
        self.proj = nn.Conv3d(dim, dim, 1)
        self.bn = nn.BatchNorm3d(dim)
        self.bn_2 = nn.BatchNorm3d(dim)
        self.conv_block = nn.Sequential(
            nn.Conv3d(in_channels=dim, out_channels=dim,
                      kernel_size=1, stride=1))

    def forward(self, x, g):
        x_ = x.clone()
        g_ = g.clone()

        multi = self.multi(x, g)

        multi = self.selection(multi)

        attention_weights = F.softmax(multi, dim=1)
        A, B = attention_weights.split(1, dim=1)

        x_att = A.expand_as(x_) * x_
        g_att = B.expand_as(g_) * g_

        x_att = x_att + x_
        g_att = g_att + g_

        x_sig = torch.sigmoid(x_att)
        g_att_2 = x_sig * g_att

        g_sig = torch.sigmoid(g_att)
        x_att_2 = g_sig * x_att

        interaction = x_att_2 * g_att_2

        projected = torch.sigmoid(self.bn(self.proj(interaction)))

        weighted = projected * x_

        y = self.conv_block(weighted)

        y = self.bn_2(y)
        return y


class Mlp(nn.Module):
    """
    MLP with depthwise convolution, activation, and dropout.
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0., linear=False):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv3d(in_features, hidden_features, 1)  # First fully connected layer (1x1 conv)
        self.dwconv = DWConvLKA(hidden_features)  # Depthwise conv layer
        self.act = act_layer()  # Activation function (GELU by default)
        self.fc2 = nn.Conv3d(hidden_features, out_features, 1)  # Second fully connected layer (1x1 conv)
        self.drop = nn.Dropout(drop)  # Dropout layer
        self.linear = linear
        if self.linear:
            self.relu = nn.ReLU(inplace=True)  # ReLU activation if linear

    def forward(self, x):
        """Forward pass."""
        x = self.fc1(x)  # Apply first fully connected layer
        if self.linear:
            x = self.relu(x)  # Apply ReLU if linear
        x = self.dwconv(x)  # Apply depthwise convolution
        x = self.act(x)  # Apply activation function
        x = self.drop(x)  # Apply dropout
        x = self.fc2(x)  # Apply second fully connected layer
        x = self.drop(x)  # Apply dropout again
        return x


class AttentionModule(nn.Module):
    """
    Applies spatial attention using convolutions for feature refinement.

    Args:
        dim (int): The input/output dimension.
    """
    def __init__(self, dim):
        super().__init__()
        self.conv0 = nn.Conv3d(dim, dim, 5, padding=2, groups=dim)  # Initial convolution
        self.conv_spatial = nn.Conv3d(dim, dim, 7, stride=1, padding=9, groups=dim, dilation=3)  # Spatial attention convolution
        self.conv1 = nn.Conv3d(dim, dim, 1)  # Final convolution

    def forward(self, x):
        """Forward pass for spatial attention."""
        u = x.clone()  # Save input for skip connection
        attn = self.conv0(x)
        attn = self.conv_spatial(attn)
        attn = self.conv1(attn)
        return u * attn  # Apply attention to input


class SpatialAttention(nn.Module):
    """
    Implements a spatial attention mechanism with gating.

    Args:
        d_model (int): The input/output dimension.
    """
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model
        self.proj_1 = nn.Conv3d(d_model, d_model, 1)  # Projection to reduce dimension
        self.activation = nn.GELU()  # Activation function
        self.spatial_gating_unit = AttentionModule(d_model)  # Attention module
        self.proj_2 = nn.Conv3d(d_model, d_model, 1)  # Projection to original dimension

    def forward(self, x):
        """Forward pass with spatial gating."""
        shortcut = x.clone()  # Save input for residual connection
        x = self.proj_1(x)
        x = self.activation(x)
        x = self.spatial_gating_unit(x)
        x = self.proj_2(x)
        return x + shortcut  # Skip connection
        

class EfficientAttention(nn.Module):
    """
    Efficient attention mechanism using 3D convolutions for key, query, and value projections.

    Args:
        in_channels (int): Input channels.
        key_channels (int): Channels for the key projection.
        value_channels (int): Channels for the value projection.
        head_count (int, optional): Number of attention heads (default is 1).
    """

    def __init__(self, in_channels, key_channels, value_channels, head_count=1):
        super().__init__()
        self.in_channels = in_channels
        self.key_channels = key_channels
        self.head_count = head_count
        self.value_channels = value_channels

        # 3D convolutions for key, query, and value projections
        self.keys = nn.Conv3d(in_channels, key_channels, 1)
        self.queries = nn.Conv3d(in_channels, key_channels, 1)
        self.values = nn.Conv3d(in_channels, value_channels, 1)
        self.reprojection = nn.Conv3d(value_channels, in_channels, 1)

    def forward(self, input_):
        """Forward pass for efficient attention."""
        n, _, d, h, w = input_.size()

        # Project inputs to keys, queries, and values
        keys = self.keys(input_).reshape((n, self.key_channels, d * h * w))
        queries = self.queries(input_).reshape(n, self.key_channels, d * h * w)
        values = self.values(input_).reshape((n, self.value_channels, d * h * w))

        # Split channels into heads
        head_key_channels = self.key_channels // self.head_count
        head_value_channels = self.value_channels // self.head_count

        attended_values = []
        for i in range(self.head_count):
            # Compute attention for each head
            key = F.softmax(keys[:, i * head_key_channels : (i + 1) * head_key_channels, :], dim=2)
            query = F.softmax(queries[:, i * head_key_channels : (i + 1) * head_key_channels, :], dim=1)
            value = values[:, i * head_value_channels : (i + 1) * head_value_channels, :]
            context = key @ value.transpose(1, 2)  # dk*dv
            attended_value = (context.transpose(1, 2) @ query).reshape(n, head_value_channels, d, h, w)  # n*dv
            attended_values.append(attended_value)

        # Concatenate heads and apply final projection
        aggregated_values = torch.cat(attended_values, dim=1)
        attention = self.reprojection(aggregated_values)

        return attention


class ChannelAttention(nn.Module):
    """
    Lightweight Channel Attention using global context (SE-style).
    Args:
        dim (int): Number of input channels.
        reduction (int): Reduction ratio for hidden layer size.
    """
    def __init__(self, dim, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(dim, dim // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(dim // reduction, dim, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (B, N, C)
        Returns:
            Tensor of shape (B, N, C) with channel-wise scaling
        """
        # Compute average over tokens N → squeeze to shape (B, C, 1)
        x_perm = x.permute(0, 2, 1)  # (B, C, N)
        y = self.avg_pool(x_perm)    # (B, C, 1)
        y = y.view(x.size(0), -1)    # (B, C)

        # Fully connected attention weights
        y = self.fc(y).unsqueeze(1)  # (B, 1, C)

        # Apply channel-wise attention
        return x * y  # (B, N, C)


class OverlapPatchEmbeddings(nn.Module):
    """
    A module that extracts overlapping patches from a 3D input tensor using a convolutional layer, followed by LayerNorm.
    Args:
        patch_size (int, optional): Size of the patches to extract. Default is 7.
        stride (int, optional): Stride of the convolution. Default is 4.
        padding (int, optional): Padding applied to the input tensor. Default is 2.
        in_ch (int, optional): Number of input channels. Default is 2.
        dim (int, optional): Output feature dimension. Default is 768.

    Forward Pass:
        The forward method applies the convolutional layer to the input tensor, flattens the spatial dimensions, 
        and normalizes the resulting embeddings before returning them along with the output dimensions.
    """

    def __init__(self, patch_size=7, stride=4, padding=2, in_ch=2, dim=768):
        super().__init__()
        self.patch_size = patch_size
        self.stride = stride
        self.padding = padding
        self.proj = nn.Conv3d(in_ch, dim, patch_size, stride, padding)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> tuple:
        px = self.proj(x)
        _, _, D, H, W = px.shape  # Dynamically get output dimensions
        fx = px.flatten(2).transpose(1, 2)  # Flatten spatial dimensions
        nfx = self.norm(fx)
        return nfx, D, H, W  # Return features and spatial dimensions

class DualTransformerBlock(nn.Module):
    """
    A dual transformer block that applies efficient attention, channel attention, and MLP transformations
    in sequence. It operates on a tensor with shape (b, D*H*W, d) and outputs a tensor of the same shape.

    Args:
        in_dim (int): The input dimension (d).
        key_dim (int): The key dimension for the attention mechanism.
        value_dim (int): The value dimension for the attention mechanism.
        head_count (int, optional): The number of attention heads (default is 1).
        token_mlp (str, optional): The type of MLP ("mix", "mix_skip", or "standard", default is "mix").
    """

    def __init__(self, in_dim, key_dim, value_dim, head_count=1, token_mlp="mix"):
        super().__init__()

        # Layer normalization and attention setup
        self.norm1 = nn.LayerNorm(in_dim)  # First normalization
        self.attn = EfficientAttention(in_channels=in_dim, key_channels=key_dim, value_channels=value_dim, head_count=head_count)  # Attention layer
        self.norm2 = nn.LayerNorm(in_dim)  # Second normalization
        self.channel_attn = ChannelAttention(in_dim)  # Channel attention layer
        self.norm3 = nn.LayerNorm(in_dim)  # Third normalization
        self.norm4 = nn.LayerNorm(in_dim)  # Fourth normalization

        # MLP configurations based on `token_mlp`
        if token_mlp == "mix":
            self.mlp1 = MixFFN(in_dim, int(in_dim * 4))
            self.mlp2 = MixFFN(in_dim, int(in_dim * 4))
        else:
            self.mlp1 = MixFFN_skip(in_dim, int(in_dim * 4))
            self.mlp2 = MixFFN_skip(in_dim, int(in_dim * 4))

    def forward(self, x: torch.Tensor, D: int, H: int, W: int) -> torch.Tensor:
        """
        Forward pass through the dual transformer block. Applies attention, channel attention, and MLP.

        Args:
            x (Tensor): The input tensor of shape (b, D*H*W, d).
            D (int): Depth dimension.
            H (int): Height dimension.
            W (int): Width dimension.

        Returns:
            Tensor: The output tensor after applying attention and MLP layers.
        """

        # Apply first normalization and reshape for attention
        norm1 = self.norm1(x)
        norm1 = rearrange(norm1, "b (d h w) c -> b c d h w", d=D, h=H, w=W)

        # Apply efficient attention
        attn = self.attn(norm1)
        attn = rearrange(attn, "b c d h w -> b (d h w) c")

        # Add attention to input and apply MLP transformations
        add1 = x + attn
        norm2 = self.norm2(add1)
        mlp1 = self.mlp1(norm2, D, H, W)

        # Apply second MLP and channel attention
        add2 = add1 + mlp1
        norm3 = self.norm3(add2)
        channel_attn = self.channel_attn(norm3)

        # Add channel attention to the result and apply final MLP
        add3 = add2 + channel_attn
        norm4 = self.norm4(add3)
        mlp2 = self.mlp2(norm4, D, H, W)

        # Final output after adding the second MLP
        mx = add3 + mlp2
        return mx


class LKABlock(nn.Module):
    """
    A block implementing Local Key Attention (LKA) and a MLP for feature transformation. 
    Used in the decoder layer to apply attention-based processing and non-linear transformations.

    Args:
        dim (int): The input dimension.
        mlp_ratio (float, optional): The ratio of the MLP hidden dimension to the input dimension (default is 4).
        drop (float, optional): Dropout rate for MLP (default is 0).
        drop_path (float, optional): Drop path rate for stochastic depth (default is 0).
        act_layer (nn.Module, optional): The activation function used in the MLP (default is `nn.GELU`).
        linear (bool, optional): Whether to use a linear activation (default is `False`).

    Methods:
        forward(x): Performs attention and MLP transformations on input `x`.

    Returns:
        Tensor: The transformed tensor after attention and MLP.
    """

    def __init__(self,
                 dim,
                 mlp_ratio=4.,
                 drop=0.,
                 drop_path=0.,
                 act_layer=nn.GELU,
                 linear=False):
        super().__init__()

        # Layer normalization for the input of attention and MLP blocks
        self.norm1 = nn.LayerNorm(dim)

        # Spatial attention mechanism for the input
        self.attn = SpatialAttention(dim)

        # Drop path for stochastic depth (optional)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        # Second layer normalization for the output of the attention block
        self.norm2 = nn.LayerNorm(dim)

        # MLP hidden dimension determined by the mlp_ratio
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        # Initialize learnable scaling factors for residual connections
        layer_scale_init_value = 1e-2
        self.layer_scale_1 = nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)
        self.layer_scale_2 = nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)

    def forward(self, x):
        """
        Forward pass through the LKA block. Applies attention and MLP transformations.

        Args:
            x (Tensor): The input tensor of shape (B, D, H, W, C).

        Returns:
            Tensor: The transformed tensor after attention and MLP operations.
        """
        # Attention block - first pass through normalization, attention, and residual connection
        y = x.permute(0, 2, 3, 4, 1)  # Rearrange to (B, D, H, W, C)
        y = self.norm1(y)  # Normalize input
        y = y.permute(0, 4, 1, 2, 3)  # Rearrange to (B, C, D, H, W)
        y = self.attn(y)  # Apply spatial attention
        y = self.layer_scale_1.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) * y  # Apply layer scaling
        y = self.drop_path(y)  # Drop path for stochastic depth
        x = x + y  # Add the residual connection

        # MLP block - second pass through normalization, MLP, and residual connection
        y = x.permute(0, 2, 3, 4, 1)  # Rearrange to (B, D, H, W, C)
        y = self.norm2(y)  # Normalize input
        y = y.permute(0, 4, 1, 2, 3)  # Rearrange to (B, C, D, H, W)
        y = self.mlp(y)  # Apply MLP transformation
        y = self.layer_scale_2.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) * y  # Apply layer scaling
        y = self.drop_path(y)  # Drop path for stochastic depth
        x = x + y  # Add the residual connection

        return x  # Return the transformed tensor


class PatchExpand(nn.Module):
    """
    A module that expands the input feature map by a factor of 2 and applies normalization.

    Args:
        input_resolution (tuple): Spatial dimensions of the input (D, H, W).
        dim (int): The input channel dimension.
        dim_scale (int, optional): The scaling factor for expansion (default is 2).
        norm_layer (nn.Module, optional): Normalization layer to apply (default is nn.LayerNorm).
    """
    def __init__(self, dim, dim_scale=2, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.expand = nn.Linear(dim, 4 * dim, bias=False) if dim_scale == 2 else nn.Identity()
        self.norm = norm_layer(dim // dim_scale)

    def forward(self, x, input_resolution):
        """
        Forward pass to expand and normalize the input tensor.

        Args:
            x (tensor): Input tensor of shape (B, D*H*W, C), where B is batch size, 
                        D, H, W are spatial dimensions, and C is the number of channels.

        Returns:
            tensor: The expanded and normalized tensor.
        """
        D, H, W = input_resolution
        x = self.expand(x)
        B, L, C = x.shape
        assert L == D * H * W, "Input feature has wrong size"
        x = x.view(B, D, H, W, C)
        x = rearrange(x, "b d h w (p1 p2 p3 c) -> b (d p1) (h p2) (w p3) c", p1=2, p2=2, p3=2, c=C // 8)
        x = x.view(B, -1, C // 8)
        x = self.norm(x.clone())
        return x

class FinalPatchExpand_X4(nn.Module):
    """
    A module that expands the input feature map by a factor of 4 and applies normalization.

    Args:
        input_resolution (tuple): Spatial dimensions of the input (D, H, W).
        dim (int): The input channel dimension.
        dim_scale (int, optional): The scaling factor for expansion (default is 4).
        norm_layer (nn.Module, optional): Normalization layer to apply (default is nn.LayerNorm).
    """
    def __init__(self, dim, dim_scale=4, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.dim_scale = dim_scale
        self.expand = nn.Linear(dim, dim * (dim_scale ** 3), bias=False)
        self.output_dim = dim
        self.norm = norm_layer(self.output_dim)

    def forward(self, x, input_resolution):
        """
        Forward pass to expand and normalize the input tensor by a factor of 4.

        Args:
            x (tensor): Input tensor of shape (B, D*H*W, C), where B is batch size, 
                        D, H, W are spatial dimensions, and C is the number of channels.

        Returns:
            tensor: The expanded and normalized tensor.
        """
        D, H, W = input_resolution
        x = self.expand(x)
        B, L, C = x.shape
        assert L == D * H * W, "Input feature has wrong size"
        x = x.view(B, D, H, W, C)
        x = rearrange(x, "b d h w (p1 p2 p3 c) -> b (d p1) (h p2) (w p3) c", p1=self.dim_scale, p2=self.dim_scale, p3=self.dim_scale, c=C // (self.dim_scale ** 3))
        x = x.view(B, -1, self.output_dim)
        x = self.norm(x.clone())
        return x
        

class MyDecoderLayerDAEFormer(nn.Module):
    """
    Decoder layer for DAEFormer, featuring multi-scale attention and transformer blocks.
    Upscales input features and applies attention mechanisms, with an optional skip connection.

    Args:
        in_out_chan (list): [input_dim, output_dim, key_dim, value_dim, x1_dim].
        head_count (int): Number of attention heads.
        token_mlp_mode (str): Mode for token MLP.
        reduction_ratio (float): Reduction ratio for attention.
        n_class (int, optional): Number of output classes (default: 3).
        norm_layer (nn.Module, optional): Normalization layer (default: nn.LayerNorm).
        is_last (bool, optional): Flag for the last decoder layer (default: False).
    """

    def __init__(self, in_out_chan, head_count, token_mlp_mode, reduction_ratio, n_class=3, norm_layer=nn.LayerNorm, is_last=False):
        super().__init__()

        dims, out_dim, key_dim, value_dim, x1_dim = in_out_chan
        self.is_last = is_last

        self.x1_linear = nn.Linear(x1_dim, out_dim)

        if not is_last:
            self.ag_attn = MultiScaleGatedAttn(dim=out_dim)
            self.ag_attn_norm = norm_layer(out_dim)
            self.layer_up = PatchExpand(dim=out_dim, dim_scale=2, norm_layer=norm_layer)
            self.last_layer = None
        else:
            self.ag_attn = MultiScaleGatedAttn(dim=out_dim)
            self.ag_attn_norm = norm_layer(out_dim)
            self.layer_up = FinalPatchExpand_X4(dim=out_dim, dim_scale=4, norm_layer=norm_layer)
            self.last_layer = nn.Conv3d(out_dim, n_class, kernel_size=1)

        self.tran_layer1 = DualTransformerBlock(in_dim=out_dim, key_dim=key_dim, value_dim=value_dim, head_count=head_count)
        self.tran_layer2 = DualTransformerBlock(in_dim=out_dim, key_dim=key_dim, value_dim=value_dim, head_count=head_count)

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv3d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x1, x2=None, input_resolution=None):
        if x2 is not None:
            # x2 shape: (B, C, D, H, W)
            b2, c2, d2, h2, w2 = x2.shape
            x2_flat = x2.view(b2, c2, -1).transpose(1, 2)  # (B, N, C)

            # Transform x1
            x1_expand = self.x1_linear(x1)  # (B, N, out_dim)

            # Reshape to 5D for attention
            x2_5d = x2  # (B, C, D, H, W)
            x1_5d = x1_expand.transpose(1, 2).view(b2, -1, d2, h2, w2)  # (B, out_dim, D, H, W)

            attn_gate = self.ag_attn(x=x2_5d, g=x1_5d)  # (B, out_dim, D, H, W)
            cat_linear_x = x1_5d + attn_gate  # (B, out_dim, D, H, W)
            cat_linear_x = cat_linear_x.permute(0, 2, 3, 4, 1)  # (B, D, H, W, C)
            cat_linear_x = self.ag_attn_norm(cat_linear_x)
            cat_linear_x = cat_linear_x.view(b2, -1, cat_linear_x.shape[-1])  # (B, N, C)

            tran_layer_1 = self.tran_layer1(cat_linear_x, d2, h2, w2)
            tran_layer_2 = self.tran_layer2(tran_layer_1, d2, h2, w2)

            out = self.layer_up(tran_layer_2, (d2, h2, w2))  # (B, N_up, C)

            if self.last_layer:
                # Convert to 5D for Conv3d
                d_up, h_up, w_up = 4 * d2, 4 * h2, 4 * w2
                out = out.view(b2, d_up, h_up, w_up, -1).permute(0, 4, 1, 2, 3)  # (B, C, D, H, W)
                out = self.last_layer(out)  # (B, n_class, D, H, W)
        else:
            if input_resolution is None:
                raise ValueError("input_resolution must be provided when x2 is None")
            out = self.layer_up(x1, input_resolution)

        return out


class MyDecoderLayerLKA(nn.Module):
    """
    A decoder layer implementing Local Key Attention (LKA) mechanism for feature processing. 
    This layer combines gated attention, LKA blocks, and patch expansion for generating the final output.
    
    Args:
        input_size (tuple): The spatial resolution of the input (D, H, W).
        in_out_chan (tuple): A tuple containing input and output channel dimensions for various stages.
        head_count (int): The number of attention heads.
        token_mlp_mode (str): The mode for the MLP layer in token processing.
        reduction_ratio (float): The reduction ratio for dimensionality reduction.
        n_class (int, optional): The number of output classes (default is 3).
        norm_layer (nn.Module, optional): The normalization layer to use (default is `nn.LayerNorm`).
        is_last (bool, optional): Whether this is the last decoder layer (default is False).
    """
    def __init__(self, in_out_chan, head_count, token_mlp_mode, reduction_ratio, n_class=3, norm_layer=nn.LayerNorm, is_last=False):
        super().__init__()
        dims, out_dim, key_dim, value_dim, x1_dim = in_out_chan
        self.is_last = is_last

        if not is_last:
            self.x1_linear = nn.Linear(x1_dim, out_dim)
            self.ag_attn = MultiScaleGatedAttn(dim=x1_dim)
            self.ag_attn_norm = nn.LayerNorm(out_dim)
            self.layer_up = PatchExpand(dim=out_dim, dim_scale=2, norm_layer=norm_layer)
            self.last_layer = None
        else:
            self.x1_linear = nn.Linear(x1_dim, out_dim)
            self.ag_attn = MultiScaleGatedAttn(dim=x1_dim)
            self.ag_attn_norm = nn.LayerNorm(out_dim)
            self.layer_up = FinalPatchExpand_X4(dim=out_dim, dim_scale=4, norm_layer=norm_layer)
            self.last_layer = nn.Conv3d(out_dim, n_class, 1)

        self.layer_lka_1 = LKABlock(dim=out_dim)
        self.layer_lka_2 = LKABlock(dim=out_dim)
        
        # Call the initialization method
        self.init_weights()

    def init_weights(self):
        """
        Initialize the weights of the layers using Xavier initialization for linear layers,
        and normal initialization for convolution and normalization layers.
        """
        for m in self.modules():
            if isinstance(m, nn.Linear):  # For linear layers
                nn.init.xavier_uniform_(m.weight)  # Xavier uniform initialization
                if m.bias is not None:
                    nn.init.zeros_(m.bias)  # Zero initialization for biases
            elif isinstance(m, nn.LayerNorm):  # For layer normalization
                nn.init.ones_(m.weight)  # Initialize weights to 1
                nn.init.zeros_(m.bias)  # Initialize bias to 0
            elif isinstance(m, nn.Conv3d):  # For convolutional layers
                nn.init.xavier_uniform_(m.weight)  # Xavier uniform initialization for convolution
                if m.bias is not None:
                    nn.init.zeros_(m.bias)  # Zero initialization for biases

    def forward(self, x1, x2=None, input_resolution=None):
        """
        Forward pass through the decoder layer. If `x2` is provided, the skip connection
        mechanism is applied; otherwise, only `x1` is processed.

        Args:
            x1 (Tensor): The first input tensor
            x2 (Tensor, optional): The second input tensor for skip connection (default: None)

        Returns:
            Tensor: The output tensor after passing through the decoder layer
        """
        if x2 is not None:
            x2 = x2.contiguous()
            b2, c2, d2, h2, w2 = x2.shape
            x2 = x2.view(b2, -1, c2)

            # Apply linear transformation to x1
            x1_expand = self.x1_linear(x1)

            # Reshape x2 and x1 for attention calculation
            x2_new = x2.view(b2, c2, d2, h2, w2)
            x1_expand = x1_expand.view(b2, c2, d2, h2, w2)

            # Apply gated attention mechanism
            attn_gate = self.ag_attn(x=x2_new, g=x1_expand)

            # Combine attention output with x1
            cat_linear_x = x1_expand + attn_gate
            cat_linear_x = cat_linear_x.permute(0, 2, 3, 4, 1)
            cat_linear_x = self.ag_attn_norm(cat_linear_x)

            # Rearrange again for the LKA blocks
            cat_linear_x = cat_linear_x.permute(0, 4, 1, 2, 3).contiguous()

            # Pass through LKA blocks
            tran_layer_1 = self.layer_lka_1(cat_linear_x)
            tran_layer_2 = self.layer_lka_2(tran_layer_1)

            # Flatten the output for the final processing layer
            tran_layer_2 = tran_layer_2.view(b2, -1, c2)
            if self.last_layer: # If the last layer exists, apply it
                out = self.layer_up(tran_layer_2, (d2, h2, w2))
                out = self.last_layer(out.view(b2, 4 * d2, 4 * h2, 4 * w2, -1).permute(0, 4, 1, 2, 3)) # If no last layer, just return the expanded output
            else:
                out = self.layer_up(tran_layer_2, (d2, h2, w2)) # If no skip connection, just apply the expansion on x1
        else:
            out = self.layer_up(x1, input_resolution)
        return out


class MiT(nn.Module):
    """
    MiT (Multi-scale Transformer) model with significantly reduced parameters while maintaining 4 stages
    """

    def __init__(self, in_dim, key_dim, value_dim, layers, head_count=1, token_mlp="mix_skip"):
        super().__init__()
        patch_sizes = [7, 3, 3, 3, 3]
        strides = [4, 2, 2, 2, 2]
        padding_sizes = [3, 1, 1, 1, 1]

        # Patch embeddings for each stage
        self.patch_embed1 = OverlapPatchEmbeddings(patch_sizes[0], strides[0], padding_sizes[0], 2, in_dim[0])
        self.patch_embed2 = OverlapPatchEmbeddings(patch_sizes[1], strides[1], padding_sizes[1], in_dim[0], in_dim[1])
        self.patch_embed3 = OverlapPatchEmbeddings(patch_sizes[2], strides[2], padding_sizes[2], in_dim[1], in_dim[2])
        self.patch_embed4 = OverlapPatchEmbeddings(patch_sizes[3], strides[3], padding_sizes[3], in_dim[2], in_dim[3])

        # Transformer blocks
        self.block1 = nn.ModuleList([DualTransformerBlock(in_dim[0], key_dim[0], value_dim[0], head_count, token_mlp) for _ in range(layers[0])])
        self.norm1 = nn.LayerNorm(in_dim[0])
        self.block2 = nn.ModuleList([DualTransformerBlock(in_dim[1], key_dim[1], value_dim[1], head_count, token_mlp) for _ in range(layers[1])])
        self.norm2 = nn.LayerNorm(in_dim[1])
        self.block3 = nn.ModuleList([DualTransformerBlock(in_dim[2], key_dim[2], value_dim[2], head_count, token_mlp) for _ in range(layers[2])])
        self.norm3 = nn.LayerNorm(in_dim[2])
        self.block4 = nn.ModuleList([DualTransformerBlock(in_dim[3], key_dim[3], value_dim[3], head_count, token_mlp) for _ in range(layers[3])])
        self.norm4 = nn.LayerNorm(in_dim[3])

    def forward(self, x: torch.Tensor) -> list:
        """
        Forward pass for MiT with 4 stages
        """
        B, C, D_in, H_in, W_in = x.shape
        outs = []

        # Stage 1
        x, D, H, W = self.patch_embed1(x)
        for blk in self.block1:
            x = blk(x, D, H, W)
        x = self.norm1(x)
        x = x.reshape(B, D, H, W, -1).permute(0, 4, 1, 2, 3).contiguous()
        outs.append(x)

        # Stage 2
        x, D, H, W = self.patch_embed2(x)
        for blk in self.block2:
            x = blk(x, D, H, W)
        x = self.norm2(x)
        x = x.reshape(B, D, H, W, -1).permute(0, 4, 1, 2, 3).contiguous()
        outs.append(x)

        # Stage 3
        x, D, H, W = self.patch_embed3(x)
        for blk in self.block3:
            x = blk(x, D, H, W)
        x = self.norm3(x)
        x = x.reshape(B, D, H, W, -1).permute(0, 4, 1, 2, 3).contiguous()
        outs.append(x)

        # Stage 4
        x, D, H, W = self.patch_embed4(x)
        for blk in self.block4:
            x = blk(x, D, H, W)
        x = self.norm4(x)
        x = x.reshape(B, D, H, W, -1).permute(0, 4, 1, 2, 3).contiguous()
        outs.append(x)

        return outs

    
class MultiScaleLight(nn.Module):
    """
    Balanced Msa2Net with moderate dimensions and 4-stage structure
    """
    def __init__(self, num_classes=3, head_count=2, reduction_ratio=2, token_mlp_mode="mix_skip"):
        super().__init__()
        # Moderately scaled dimensions
        dims = [48, 96, 192, 384]
        key_dim = [48, 96, 192, 384]
        value_dim = [48, 96, 192, 384]
        layers = [1, 1, 1, 1]

        self.backbone = MiT(
            in_dim=dims,
            key_dim=key_dim,
            value_dim=value_dim,
            layers=layers,
            head_count=head_count,
            token_mlp=token_mlp_mode
        )

        in_out_chan = [
            [48, 48, 48, 48, 48],
            [96, 96, 96, 96, 96],
            [192, 192, 192, 192, 192],
            [384, 384, 384, 384, 384]
        ]

        self.decoder_3 = MyDecoderLayerDAEFormer(in_out_chan[3], head_count, token_mlp_mode, reduction_ratio, n_class=num_classes)
        self.decoder_2 = MyDecoderLayerDAEFormer(in_out_chan[2], head_count, token_mlp_mode, reduction_ratio, n_class=num_classes)
        self.decoder_1 = MyDecoderLayerLKA(in_out_chan[1], head_count, token_mlp_mode, reduction_ratio, n_class=num_classes)
        self.decoder_0 = MyDecoderLayerLKA(in_out_chan[0], head_count, token_mlp_mode, reduction_ratio, n_class=num_classes, is_last=True)

    def forward(self, x):
        """
        Forward pass for balanced Msa2Net with 4 stages
        """
        # Handle single-channel input
        if x.size()[1] == 1:
            x = x.repeat(1, 3, 1, 1, 1)

        output_enc = self.backbone(x)
        b, c, d, h, w = output_enc[3].shape

        tmp_3 = self.decoder_3(
            output_enc[3].permute(0, 2, 3, 4, 1).view(b, -1, c),
            input_resolution=(d, h, w)
        )

        tmp_2 = self.decoder_2(
            tmp_3,
            output_enc[2],
            input_resolution=(output_enc[2].shape[2], output_enc[2].shape[3], output_enc[2].shape[4])
        )
        tmp_1 = self.decoder_1(
            tmp_2,
            output_enc[1],
            input_resolution=(output_enc[1].shape[2], output_enc[1].shape[3], output_enc[1].shape[4])
        )
        tmp_0 = self.decoder_0(
            tmp_1,
            output_enc[0],
            input_resolution=(output_enc[0].shape[2], output_enc[0].shape[3], output_enc[0].shape[4])
        )

        return tmp_0

    
    
class SpatialTransformer(nn.Module):
    """
    N-Dimensional Spatial Transformer for warping images or volumes using a displacement field.

    This module performs spatial transformation by applying a flow (displacement field) to an input
    image or volume. It dynamically constructs a sampling grid, adds the flow to compute new coordinates,
    normalizes them, and uses `grid_sample` for interpolation.

    Args:
        mode (str): Interpolation mode to be used in grid sampling.
                    Options: 'bilinear', 'nearest', or 'bicubic' (for 4D inputs only).
                    Default is 'bilinear'.

    Forward Args:
        src (torch.Tensor): The input tensor to be warped.
                            Shape must be (B, C, H, W) for 2D or (B, C, D, H, W) for 3D.

        flow (torch.Tensor): The displacement field indicating where to sample from `src`.
                             Shape must be (B, 2, H, W) for 2D or (B, 3, D, H, W) for 3D.
                             The last dimension size must match the spatial dimensions of `src`.

    Returns:
        torch.Tensor: The warped image or volume, same shape as `src`.
    """

    def __init__(self, mode='bilinear'):
        super().__init__()
        self.mode = mode

    def forward(self, src, flow):
        # Get spatial dimensions and number of dimensions
        shape = flow.shape[2:]  # e.g., (D, H, W) or (H, W)
        dim = len(shape)
        device = flow.device
        dtype = flow.dtype

        # Create normalized coordinate grid
        coords = torch.meshgrid([torch.arange(0, s, device=device, dtype=dtype) for s in shape], indexing='ij')
        grid = torch.stack(coords)  # Shape: (N, ...)
        grid = grid.unsqueeze(0).expand(flow.shape[0], -1, *shape)  # Shape: (B, N, ...)

        # Apply displacement field
        new_locs = grid + flow

        # Normalize coordinates to [-1, 1] range
        for i in range(dim):
            new_locs[:, i] = 2.0 * (new_locs[:, i] / (shape[i] - 1)) - 1.0

        # Rearrange dimensions to match grid_sample format
        if dim == 2:
            new_locs = new_locs.permute(0, 2, 3, 1)      # (B, H, W, 2)
            new_locs = new_locs[..., [1, 0]]             # Convert to (x, y)
        elif dim == 3:
            new_locs = new_locs.permute(0, 2, 3, 4, 1)   # (B, D, H, W, 3)
            new_locs = new_locs[..., [2, 1, 0]]          # Convert to (x, y, z)

        # Warp the source image using the flow field
        return F.grid_sample(src, new_locs, mode=self.mode, align_corners=True)


class NestedMorph(nn.Module):
    def __init__(self, inshape, use_gpu=False):
        super().__init__()
        self.unet = MultiScaleLight(num_classes=1)  # Reduced output channels
        ndims = len(inshape)
        assert ndims in [1, 2, 3], f'ndims should be 1, 2, or 3. found: {ndims}'

        self.flow = nn.Conv3d(1, ndims, kernel_size=3, padding=1)  # Adjusted input channels to match UNet output
        self.flow.weight = nn.Parameter(Normal(0, 1e-5).sample(self.flow.weight.shape))
        self.flow.bias = nn.Parameter(torch.zeros(self.flow.bias.shape))

        self.spatial_transform = SpatialTransformer()

        if use_gpu:
            self.cuda()

    def forward(self, inputs):
        source, tar = inputs
        x = torch.cat((source, tar), dim=1)
        unet_output = self.unet(x)
        flow_field = self.flow(unet_output)
        registered_image = self.spatial_transform(source, flow_field)
        return registered_image, flow_field
    
'''
MSA-2Net

A big thank you to the authors of MSA-2Net (https://github.com/xmindflow/MSA-2Net). 
Their work has been a major source of inspiration and motivation for the development of this project.

Original paper:
Ghorbani Kolahi, S., Kamal Chaharsooghi, S., Khatibi, T., Bozorgpour, A., Azad, R., Heidari, M., Hacihaliloglu, I. and Merhof, D. (2024).
MSA2Net: Multi-scale Adaptive Attention-guided Network for Medical Image Segmentation
British Machine Vision Conference, 2024.
'''
