import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.PReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.PReLU()
        )

    def forward(self, x):
        return self.conv(x)

class ResidualBlock(nn.Module):
    def __init__(self, in_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = ConvBlock(in_channels, in_channels)
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=1)

    def forward(self, x):
        residual = self.conv2(x)
        x = self.conv1(x)
        x += residual
        return x


class FusionBlock(nn.Module):
    def __init__(self, num_features, num_inputs):
        super(FusionBlock, self).__init__()
        self.num_features = num_features
        self.weights = nn.Parameter(torch.ones(num_inputs, dtype=torch.float32), requires_grad=True)
        self.conv1x1_list = nn.ModuleList(
            [nn.Conv2d(num_features, num_features, kernel_size=1) for _ in range(num_inputs)])
        self.softmax = nn.Softmax(dim=0)
        self.fusion_conv = nn.Conv2d(num_features, num_features, kernel_size=1)

    def forward(self, *inputs):
        # Compute weighted average
        weighted_features = [w * conv1x1(f) for w, conv1x1, f in
                             zip(self.softmax(self.weights), self.conv1x1_list, inputs)]
        weighted_sum = sum(weighted_features)

        # Compute the high-level feature map
        F_high = weighted_features[0]  # Assuming the first feature map is the high-level one
        F_other = weighted_sum - F_high

        # Attention mechanism
        attention = F_other * torch.sigmoid(F_high)

        # Fusion of the feature maps
        F_fusion = self.fusion_conv(attention)

        return F_fusion

class ModifiedUNet(nn.Module):
    def __init__(self):
        super(ModifiedUNet, self).__init__()
        # Encoder blocks
        self.encoder_blocks = nn.ModuleList([
            ConvBlock(3, 32),
            ConvBlock(32, 64),
            ConvBlock(64, 128),
            ConvBlock(128, 256)
        ])
        # Fusion blocks for each level of the encoder
        self.fusion_blocks = nn.ModuleList([
            FusionBlock(32, 4),
            FusionBlock(64, 4),
            FusionBlock(128, 4),
            FusionBlock(256, 4)
        ])
        # Decoder blocks
        self.decoder_blocks = nn.ModuleList([
            ConvBlock(256, 128),
            ConvBlock(128, 64),
            ConvBlock(64, 32)
        ])
        # Upsampling layers
        self.upsample_layers = nn.ModuleList([
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        ])
        # Final output convolution
        self.final_conv = nn.Conv2d(32, 1, kernel_size=1)
        # Pooling layer for downsampling in the encoder
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, *inputs):
        # Check the initial inputs
        print(f"Initial inputs shapes: {[x.shape for x in inputs]}")

        # Encoder path
        enc_features = []
        for idx, block in enumerate(self.encoder_blocks):
            inputs = [block(x) for x in inputs]
            print(f"After Encoder Block {idx}, shapes: {[x.shape for x in inputs]}")
            if idx < len(self.encoder_blocks) - 1:
                # Save the current features for the skip connections
                enc_features.append(inputs)
                # Downsample for the next encoder block
                inputs = [self.pool(x) for x in inputs]

        # At this point, 'inputs' should be a list of feature maps from the last encoder block

        # Now we should apply each fusion block to its corresponding set of feature maps
        for idx, fusion_block in enumerate(self.fusion_blocks):
            # Fuse the feature maps; the result is a single feature map
            f_combined = fusion_block(*inputs)
            # 'f_combined' is now a single tensor

            # If not the last block, prepare inputs for the next fusion block
            if idx < len(self.fusion_blocks) - 1:
                # Upsample
                upsampled_input = self.upsample_layers[idx](f_combined)
                # Prepare for concatenation with skip connections from the encoder
                # We take the corresponding feature maps from 'enc_features' which is a list of lists of tensors
                inputs = [torch.cat((upsampled_input, enc_feature[idx]), dim=1)
                          for enc_feature in enc_features[-(idx + 1)]]
                # After concatenation, 'inputs' should be back to being a list of tensors
            else:
                # If it's the last fusion block, we only have one tensor to work with
                inputs = [f_combined]

        # 'inputs' is now a list with a single tensor that went through all fusion blocks
        # We pass it through each decoder block
        for decoder_block in self.decoder_blocks:
            inputs = [decoder_block(inputs[0])]

        # Final convolution to get to the output
        output = self.final_conv(inputs[0])
        return output

if __name__ == '__main__':
    model = ModifiedUNet()
    # Create four separate input tensors with the shape [batch_size, num_channels, height, width]
    input_tensors = [torch.randn(1, 3, 256, 256) for _ in range(4)]
    output = model(*input_tensors)

    print(output.shape)
