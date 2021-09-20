"""
Implements image encoders
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import encoder_utils as util
from custom_encoder import ConvEncoder
import torch.autograd.profiler as profiler
import pdb

class SpatialEncoder(nn.Module):
    """
    2D (Spatial/Pixel-aligned/local) image encoder
    """

    def __init__(
        self,
        backbone="resnet34",
        pretrained=True,
        num_layers=4,
        index_interp="bilinear",
        index_padding="border",
        upsample_interp="bilinear",
        feature_scale=1.0,
        use_first_pool=True,
        norm_type="batch",
    ):
        """
        :param backbone Backbone network. Either custom, in which case
        model.custom_encoder.ConvEncoder is used OR resnet18/resnet34, in which case the relevant
        model from torchvision is used
        :param num_layers number of resnet layers to use, 1-5
        :param pretrained Whether to use model weights pretrained on ImageNet
        :param index_interp Interpolation to use for indexing
        :param index_padding Padding mode to use for indexing, border | zeros | reflection
        :param upsample_interp Interpolation to use for upscaling latent code
        :param feature_scale factor to scale all latent by. Useful (<1) if image
        is extremely large, to fit in memory.
        :param use_first_pool if false, skips first maxpool layer to avoid downscaling image
        features too much (ResNet only)
        :param norm_type norm type to applied; pretrained model must use batch
        """
        super().__init__()

        if norm_type != "batch":
            assert not pretrained

        self.use_custom_resnet = backbone == "custom"
        self.feature_scale = feature_scale
        self.use_first_pool = use_first_pool
        norm_layer = util.get_norm_layer(norm_type)

        if self.use_custom_resnet:
            print("WARNING: Custom encoder is experimental only")
            print("Using simple convolutional encoder")
            self.model = ConvEncoder(3, norm_layer=norm_layer)
            self.latent_size = self.model.dims[-1]
        else:
            print("Using torchvision", backbone, "encoder")
            self.model = getattr(torchvision.models, backbone)(
                pretrained=pretrained, norm_layer=norm_layer
            )
            # Following 2 lines need to be uncommented for older configs
            self.model.fc = nn.Sequential()
            self.model.avgpool = nn.Sequential()
            self.latent_size = [0, 64, 128, 256, 512, 1024][num_layers]

        self.num_layers = num_layers
        self.index_interp = index_interp
        self.index_padding = index_padding
        self.upsample_interp = upsample_interp
        self.register_buffer("latent", torch.empty(1, 1, 1, 1))#, persistent=False)
        self.register_buffer(
            "latent_scaling", torch.empty(2, dtype=torch.float32)#, persistent=False
        )
        # self.latent (B, L, H, W)

    def index(self, uv, cam_z=None, image_size=(), z_bounds=None):
        """
        Get pixel-aligned image features at 2D image coordinates
        :param uv (B, N, 2) image points (x,y)
        :param cam_z ignored (for compatibility)
        :param image_size image size, either (width, height) or single int.
        if not specified, assumes coords are in [-1, 1]
        :param z_bounds ignored (for compatibility)
        :return (B, L, N) L is latent size
        """
        with profiler.record_function("encoder_index"):
            if uv.shape[0] == 1 and self.latent.shape[0] > 1:
                uv = uv.expand(self.latent.shape[0], -1, -1)

            with profiler.record_function("encoder_index_pre"):
                if len(image_size) > 0:
                    if len(image_size) == 1:
                        image_size = (image_size, image_size)
                    scale = self.latent_scaling / image_size
                    uv = uv * scale - 1.0

            uv = uv.unsqueeze(2)  # (B, N, 1, 2)
            samples = F.grid_sample(
                self.latent,
                uv,
                align_corners=True,
                mode=self.index_interp,
                padding_mode=self.index_padding,
            )
            return samples[:, :, :, 0]  # (B, C, N)

    def forward(self, x):
        """
        For extracting ResNet's features.
        :param x image (B, C, H, W)
        :return latent (B, latent_size, H, W)
        """
        if self.feature_scale != 1.0:
            x = F.interpolate(
                x,
                scale_factor=self.feature_scale,
                mode="bilinear" if self.feature_scale > 1.0 else "area",
                align_corners=True if self.feature_scale > 1.0 else None,
                recompute_scale_factor=True,
            )
        x = x.to(device=self.latent.device)

        if self.use_custom_resnet:
            self.latent = self.model(x)
        else:
            x = self.model.conv1(x)
            x = self.model.bn1(x)
            x = self.model.relu(x)

            latents = [x]
            if self.num_layers > 1:
                if self.use_first_pool:
                    x = self.model.maxpool(x)
                x = self.model.layer1(x)
                latents.append(x)
            if self.num_layers > 2:
                x = self.model.layer2(x)
                latents.append(x)
            if self.num_layers > 3:
                x = self.model.layer3(x)
                latents.append(x)
            if self.num_layers > 4:
                x = self.model.layer4(x)
                latents.append(x)

            self.latents = latents
            align_corners = None if self.index_interp == "nearest " else True
            latent_sz = latents[0].shape[-2:]
            for i in range(len(latents)):
                latents[i] = F.interpolate(
                    latents[i],
                    latent_sz,
                    mode=self.upsample_interp,
                    align_corners=align_corners,
                )
            self.latent = torch.cat(latents, dim=1)
        self.latent_scaling[0] = self.latent.shape[-1]
        self.latent_scaling[1] = self.latent.shape[-2]
        self.latent_scaling = self.latent_scaling / (self.latent_scaling - 1) * 2.0
        return self.latent

    @classmethod
    def from_conf(cls, conf):
        return cls(
            conf.get_string("backbone"),
            pretrained=conf.get_bool("pretrained", True),
            num_layers=conf.get_int("num_layers", 4),
            index_interp=conf.get_string("index_interp", "bilinear"),
            index_padding=conf.get_string("index_padding", "border"),
            upsample_interp=conf.get_string("upsample_interp", "bilinear"),
            feature_scale=conf.get_float("feature_scale", 1.0),
            use_first_pool=conf.get_bool("use_first_pool", True),
        )


class ImageEncoder(nn.Module):
    """
    Global image encoder
    """

    def __init__(self, backbone="resnet34", pretrained=True, latent_size=128):
        """
        :param backbone Backbone network. Assumes it is resnet*
        e.g. resnet34 | resnet50
        :param num_layers number of resnet layers to use, 1-5
        :param pretrained Whether to use model pretrained on ImageNet
        """
        super().__init__()
        self.model = getattr(torchvision.models, backbone)(pretrained=pretrained)
        self.model.fc = nn.Sequential()
        self.register_buffer("latent", torch.empty(1, 1))#, persistent=False)
        # self.latent (B, L)
        self.latent_size = latent_size
        if latent_size != 512:
            self.fc = nn.Linear(512, latent_size)

    def index(self, uv, cam_z=None, image_size=(), z_bounds=()):
        """
        Params ignored (compatibility)
        :param uv (B, N, 2) only used for shape
        :return latent vector (B, L, N)
        """
        return self.latent.unsqueeze(-1).expand(-1, -1, uv.shape[1])

    def forward(self, x):
        """
        For extracting ResNet's features.
        :param x image (B, C, H, W)
        :return latent (B, latent_size)
        """
        x = x.to(device=self.latent.device)
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)

        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)

        x = self.model.avgpool(x)
        x = torch.flatten(x, 1)

        if self.latent_size != 512:
            x = self.fc(x)

        self.latent = x  # (B, latent_size)
        return self.latent

    @classmethod
    def from_conf(cls, conf):
        return cls(
            conf.get_string("backbone"),
            pretrained=conf.get_bool("pretrained", True),
            latent_size=conf.get_int("latent_size", 128),
        )


class CoarseUNet(nn.Module):
    def __init__(
            self,
            in_channels=128,
            out_channels=3,
            num_encoder_filters=[256, 256, 256, 512, 512, 512],
            num_decoder_filters=[512, 512, 512, 256, 256, 256],
            num_last_filters=[128, 64, 32],
            encoder_kernel_size=4,
            encoder_kernel_stride=2,
            decoder_kernel_size=3,
            decoder_kernel_stride=1,
            padding=False,
            batch_norm=False,
            up_mode='upconv',
            bn_momentum=0.9):
        # TODO(vfragoso): The documentation is outdated, please update.
        """
        Implementation of
        U-Net: Convolutional Networks for Biomedical Image Segmentation
        (Ronneberger et al., 2015)
        https://arxiv.org/abs/1505.04597

        Using the default arguments will yield the exact version used
        in the original paper

        Args:
            in_channels (int): number of input channels
            out_channels (int): number of output channels
            depth (int): depth of the network
            wf (int): number of filters in the first layer is 2**wf
            padding (bool): if True, apply padding such that the input shape
                            is the same as the output.
                            This may introduce artifacts
            batch_norm (bool): Use BatchNorm after layers with an
                               activation function
            up_mode (str): one of 'upconv' or 'upsample'.
                           'upconv' will use transposed convolutions for
                           learned upsampling.
                           'upsample' will use bilinear upsampling.
        """
        super(CoarseUNet, self).__init__()
        assert up_mode in ('upconv', 'upsample', 'nearest')
        self.padding = padding

        down_path_depth = len(num_encoder_filters)
        up_path_depth = len(num_decoder_filters)
        assert down_path_depth > 1, "Invalid number of encoder filters"
        assert down_path_depth == up_path_depth, \
            "Invalid number of decoder filters"
        self.depth = up_path_depth + 1

        prev_channels = in_channels
        self.down_path = nn.ModuleList()

        # Building down-path.
        for i in range(len(num_encoder_filters)):
            num_output_channels = num_encoder_filters[i]
            self.down_path.append(
                UNetConvBlock(prev_channels,
                              num_output_channels,
                              padding,
                              batch_norm,
                              kernel_size=encoder_kernel_size,
                              stride=encoder_kernel_stride)
            )
            prev_channels = num_output_channels

        # Building up-path.
        concat_dimensions = [in_channels] + num_encoder_filters[0:up_path_depth - 1]
        self.up_path = nn.ModuleList()
        for i in range(len(num_decoder_filters)):
            num_output_channels = num_decoder_filters[i]
            # prev_channels += num_encoder_filters[-i - 1]
            prev_channels += concat_dimensions[-i - 1]
            self.up_path.append(
                UNetUpBlock(prev_channels,
                            num_output_channels,
                            up_mode,
                            padding,
                            batch_norm,
                            kernel_size=decoder_kernel_size,
                            stride=decoder_kernel_stride)
            )
            prev_channels = num_output_channels

        # Building last layer of CNNs.
        last = list()
        for i, output_size in enumerate(num_last_filters):
            non_linearity_op = nn.ReLU()
            last.append(
                nn.Sequential(
                    nn.Conv2d(prev_channels,
                              output_size,
                              kernel_size=3,
                              padding=int(padding)),
                    non_linearity_op,
                    nn.BatchNorm2d(output_size, momentum=bn_momentum)))
            prev_channels = output_size

        # Last layer.
        non_linearity_op = nn.Tanh()
        last.append(
            nn.Sequential(
                nn.Conv2d(prev_channels,
                          out_channels,
                          kernel_size=3,
                          padding=int(padding)),
                non_linearity_op))
            
        self.last = nn.Sequential(*last)
            

    def forward(self, x):
        blocks = list()
        # Down path.
        for i, down in enumerate(self.down_path):
            # Keep the input before it gets downsampled.
            blocks.append(x)

            # Convolve and downsample.
            x = down(x)

            # Process decoding part except bottleneck.
            if i != len(self.down_path) - 1:
                x = F.max_pool2d(x, 1)
        
        # Up path.
        for i, up in enumerate(self.up_path):
            # pdb.set_trace()
            x = up(x, blocks[-i - 1])

        # Last conv module.
        x = self.last(x)
        return x


class UNetConvBlock(nn.Module):
    def __init__(self,
                 in_size,
                 out_size,
                 padding,
                 batch_norm,
                 kernel_size=3,
                 stride=1,
                 bn_momentum=0.9):
        super(UNetConvBlock, self).__init__()
        block = list()

        block.append(nn.Conv2d(in_size,
                               out_size,
                               kernel_size=kernel_size,
                               stride=stride,
                               padding=int(padding)))
        block.append(nn.ReLU())
        if batch_norm:
            block.append(nn.BatchNorm2d(out_size, momentum=bn_momentum))

        self.block = nn.Sequential(*block)

    def forward(self, x):
        out = self.block(x)
        return out


class UNetUpBlock(nn.Module):
    def __init__(self,
                 in_size,
                 out_size,
                 up_mode,
                 padding,
                 batch_norm,
                 kernel_size=3,
                 stride=1):
        super(UNetUpBlock, self).__init__()
        if up_mode == 'upconv':
            self.up = nn.ConvTranspose2d(in_size, out_size, kernel_size=2, stride=2)
        elif up_mode == 'upsample':
            self.up = nn.Sequential(
                nn.Upsample(mode='upsample', scale_factor=2),
                nn.Conv2d(in_size, out_size, kernel_size=1),
            )
        elif up_mode == 'nearest':
            self.up = nn.Upsample(mode='nearest', scale_factor=2)

        self.conv_block = UNetConvBlock(in_size,
                                        out_size,
                                        padding,
                                        batch_norm,
                                        kernel_size,
                                        stride)

    def center_crop(self, layer, target_size):
        _, _, layer_height, layer_width = layer.size()
        diff_y = (layer_height - target_size[0]) // 2
        diff_x = (layer_width - target_size[1]) // 2
        return layer[
            :, :, diff_y : (diff_y + target_size[0]), diff_x : (diff_x + target_size[1])
        ]

    def forward(self, x, bridge):
        up = self.up(x)
        crop1 = self.center_crop(bridge, up.shape[2:])
        out = torch.cat([up, crop1], 1)
        out = self.conv_block(out)

        return out


"""
Simple UNet demo
@author: ptrblck
"""

class BaseConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding,
                 stride):
        super(BaseConv, self).__init__()

        self.act = nn.ReLU()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, padding,
                               stride)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size,
                               padding, stride)

    def forward(self, x):
        x = self.act(self.conv1(x))
        x = self.act(self.conv2(x))
        return x


class DownConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding,
                 stride):
        super(DownConv, self).__init__()

        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.conv_block = BaseConv(in_channels, out_channels, kernel_size,
                                   padding, stride)

    def forward(self, x):
        x = self.pool1(x)
        x = self.conv_block(x)
        return x


class UpConv(nn.Module):
    def __init__(self, in_channels, in_channels_skip, out_channels,
                 kernel_size, padding, stride):
        super(UpConv, self).__init__()

        self.conv_trans1 = nn.ConvTranspose2d(
            in_channels, in_channels, kernel_size=2, padding=0, stride=2)
        self.conv_block = BaseConv(
            in_channels=in_channels + in_channels_skip,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=padding,
            stride=stride)

    def forward(self, x, x_skip):
        x = self.conv_trans1(x)
        x = torch.cat((x, x_skip), dim=1)
        x = self.conv_block(x)
        return x


class UNet(nn.Module):
    def __init__(self, in_channels, out_channels, n_class, kernel_size,
                 padding, stride):
        super(UNet, self).__init__()

        self.init_conv = BaseConv(in_channels, out_channels, kernel_size,
                                  padding, stride)

        self.down1 = DownConv(out_channels, 2 * out_channels, kernel_size,
                              padding, stride)

        self.down2 = DownConv(2 * out_channels, 4 * out_channels, kernel_size,
                              padding, stride)

        self.down3 = DownConv(4 * out_channels, 8 * out_channels, kernel_size,
                              padding, stride)

        self.up3 = UpConv(8 * out_channels, 4 * out_channels, 4 * out_channels,
                          kernel_size, padding, stride)

        self.up2 = UpConv(4 * out_channels, 2 * out_channels, 2 * out_channels,
                          kernel_size, padding, stride)

        self.up1 = UpConv(2 * out_channels, out_channels, out_channels,
                          kernel_size, padding, stride)

        self.out = nn.Conv2d(out_channels, n_class, kernel_size, padding, stride)

    def forward(self, x):
        # Encoder
        x = self.init_conv(x)
        x1 = self.down1(x)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        # Decoder
        x_up = self.up3(x3, x2)
        x_up = self.up2(x_up, x1)
        x_up = self.up1(x_up, x)
        x_out = F.log_softmax(self.out(x_up), 1)
        return x_out


if __name__ == "__main__":
	# encoder = CoarseUNet(in_channels=3, up_mode='nearest').cuda()
	x = torch.rand(1,3,1512,2016).cuda()
	# x = torch.rand(1,3,1024,2048).cuda()
	encoder = SpatialEncoder().cuda()
	y = encoder(x)
	# projected points from target to source
	uv = torch.tensor([[1500,2000],[750,1000],[100,50]]).unsqueeze(0).cuda()

	features = encoder.index(uv, None, torch.tensor([1512,2016]).cuda())
	# y[:,:10,int(1500/2),int(2000/2)] == features[0,:10,0]
	# y[:,:10,int(750/2),int(1000/2)] == features[0,:10,1]

	# maintains size
	# model = UNet(in_channels=3,
 #             out_channels=3,
 #             n_class=3,
 #             kernel_size=3,
 #             padding=1,
 #             stride=1).cuda()

	# y = model(x)
	pdb.set_trace()
