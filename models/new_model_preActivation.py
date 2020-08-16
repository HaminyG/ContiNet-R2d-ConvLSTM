import torch
import torch.nn as nn
from models.convlstm_good_v2 import ConvLSTM
import torch.nn.functional as F
class Conv3DSimple(nn.Conv3d):
    def __init__(self,
                 in_planes,
                 out_planes,
                 midplanes=None,
                 stride=1,
                 padding=1):

        super(Conv3DSimple, self).__init__(
            in_channels=in_planes,
            out_channels=out_planes,
            kernel_size=(3, 3, 3),
            stride=stride,
            padding=padding,
            bias=False)

    @staticmethod
    def get_downsample_stride(stride):
        return (stride, stride, stride)



class Conv2Plus1D(nn.Module):

    def __init__(self,
                 in_planes,
                 out_planes,
                 midplanes,
                 stride=1,
                 padding=1):
        super(Conv2Plus1D, self).__init__()
        self.bn1 = nn.BatchNorm3d(in_planes)
        self.spatio=nn.Conv3d(in_planes, midplanes, kernel_size=(1, 3, 3),
                      stride=(1, stride, stride), padding=(0, padding, padding),
                      bias=False)

        self.relu=nn.ReLU(inplace=True)
        self.temp=ConvLSTM(midplanes, out_planes, (3,3), 1, batch_first=True, bias=True, return_all_layers=False).cuda()
        #self.temp=nn.Conv3d(midplanes, out_planes, kernel_size=(3, 1, 1),
                      #stride=(stride, 1, 1), padding=(padding, 0, 0),
                      #ias=False)

    def forward(self,x):
        output = self.bn1(x)
        output = self.relu(output)
        output= self.spatio(output)
        output=output.permute(0,2,1,3,4)
        output=self.temp(output)
        output=output[0]
        output=output.permute(0,2,1,3,4)
        return output

    @staticmethod
    def get_downsample_stride(stride):
        return (1, stride, stride)


class Conv3DNoTemporal(nn.Conv3d):

    def __init__(self,
                 in_planes,
                 out_planes,
                 midplanes=None,
                 stride=1,
                 padding=1):

        super(Conv3DNoTemporal, self).__init__(
            in_channels=in_planes,
            out_channels=out_planes,
            kernel_size=(1, 3, 3),
            stride=(1, stride, stride),
            padding=(0, padding, padding),
            bias=False)

    @staticmethod
    def get_downsample_stride(stride):
        return (1, stride, stride)


class BasicBlock(nn.Module):

    expansion = 1

    def __init__(self, inplanes, planes, conv_builder, stride=1, downsample=None):
        midplanes = (inplanes * planes * 3 * 3 * 3) // (inplanes * 3 * 3 + 3 * planes)

        super(BasicBlock, self).__init__()
        self.conv1 = nn.Sequential(
            nn.BatchNorm3d(inplanes),
            nn.ReLU(inplace=True),
            conv_builder(inplanes, planes, midplanes, stride)

        )
        self.conv2 = nn.Sequential(
            nn.BatchNorm3d(planes),
            nn.ReLU(inplace=True),
            conv_builder(planes, planes, midplanes)

        )

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)

        out = self.conv2(out)


        if self.downsample is not None:
            residual = self.downsample(x)


        out += residual
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, conv_builder, stride=1, downsample=None):

        super(Bottleneck, self).__init__()
        midplanes = (inplanes * planes * 3 * 3 * 3) // (inplanes * 3 * 3 + 3 * planes)

        # 1x1x1
        self.conv1 = nn.Sequential(
            nn.BatchNorm3d(planes),
            nn.ReLU(inplace=True),
            nn.Conv3d(inplanes, planes, kernel_size=1, bias=False)

        )
        # Second kernel
        self.conv2 = nn.Sequential(
            nn.BatchNorm3d(planes),
            nn.ReLU(inplace=True),
            conv_builder(planes, planes, midplanes, stride),

        )

        # 1x1x1
        self.conv3 = nn.Sequential(
            nn.BatchNorm3d(planes * self.expansion),
            nn.ReLU(inplace=True),
            nn.Conv3d(planes, planes * self.expansion, kernel_size=1, bias=False),

        )

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual


        return out


class BasicStem(nn.Sequential):
    """The default conv-batchnorm-relu stem
    """
    def __init__(self):
        super(BasicStem, self).__init__(
            nn.Conv3d(3, 64, kernel_size=(3, 7, 7), stride=(1, 2, 2),#kernel_size=(3,7,7)
                      padding=(1, 3, 3), bias=False),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True))


class R2Plus1dStem(nn.Sequential):
    """R(2+1)D stem is different than the default one as it uses separated 3D convolution
    """
    def __init__(self):
        super(R2Plus1dStem, self).__init__(
            nn.Conv3d(3, 45, kernel_size=(1, 7, 7),#kernel_size=(1,7,7)
                      stride=(1, 2, 2), padding=(0, 3, 3),
                      bias=False),
            nn.BatchNorm3d(45),
            nn.ReLU(inplace=True),
            nn.Conv3d(45, 64, kernel_size=(3, 1, 1),
                      stride=(1, 1, 1), padding=(1, 0, 0),
                      bias=False),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True))


class VideoResNet(nn.Module):

    def __init__(self, block, conv_makers, layers,
                 stem, num_classes=100,
                 zero_init_residual=False):
        """Generic resnet video generator.

        Args:
            block (nn.Module): resnet building block
            conv_makers (list(functions)): generator function for each layer
            layers (List[int]): number of blocks per layer
            stem (nn.Module, optional): Resnet stem, if None, defaults to conv-bn-relu. Defaults to None.
            num_classes (int, optional): Dimension of the final FC layer. Defaults to 400.
            zero_init_residual (bool, optional): Zero init bottleneck residual BN. Defaults to False.
        """
        super(VideoResNet, self).__init__()
        self.inplanes = 64

        self.stem = stem()

        self.layer1 = self._make_layer(block, conv_makers[0], 64, layers[0], stride=1)
        self.layer2 = self._make_layer(block, conv_makers[1], 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, conv_makers[2], 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, conv_makers[3], 512, layers[3], stride=2)
        self.conv1 = nn.Conv3d(64, 512, kernel_size=1, bias=False)
        self.conv2 = nn.Conv3d(128, 512, kernel_size=1, bias=False)
        self.conv3 = nn.Conv3d(256, 512, kernel_size=1, bias=False)
        self.conv4 = nn.Conv3d(512, 512, kernel_size=1, bias=False)
        self.avgpool = nn.AdaptiveAvgPool3d((16, 1, 1))
        self.avgpool1 = nn.AdaptiveAvgPool3d((2, 1, 1))
        self.fc = nn.Linear(4096, num_classes)


        # init weights
        self._initialize_weights()

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)

    def forward(self, x):
        x = self.stem(x)

        x = self.layer1(x)
        #output1=F.relu(self.conv1(x))
        output1=self.conv1(x)
        output1 = self.avgpool(output1)

        x = self.layer2(x)
        #output2=F.relu(self.conv2(x))
        output2 = self.conv2(x)
        output2 = self.avgpool(output2)

        x = self.layer3(x)
        #output3=F.relu(self.conv3(x))
        output3 = self.conv3(x)
        output3 = self.avgpool(output3)

        x = self.layer4(x)
        #output4=F.relu(self.conv4(x))
        output4 = self.conv4(x)
        output4 = self.avgpool(output4)

        final = torch.cat((output1, output2, output3, output4), dim=1)

        x = self.avgpool1(final)
        # Flatten the layer to fc
        x = x.flatten(1)
        x = self.fc(x)


        return x

    def _make_layer(self, block, conv_builder, planes, blocks, stride=1):
        downsample = None

        if stride != 1 or self.inplanes != planes * block.expansion:
            ds_stride = conv_builder.get_downsample_stride(stride)
            downsample = nn.Sequential(
                nn.Conv3d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=ds_stride, bias=False),
                nn.BatchNorm3d(planes * block.expansion)
            )
        layers = []
        layers.append(block(self.inplanes, planes, conv_builder, stride, downsample))

        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, conv_builder))

        return nn.Sequential(*layers)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                        nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


def _video_resnet(arch, pretrained=False, progress=True, **kwargs):
    model = VideoResNet(**kwargs)

    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress)
        model.load_state_dict(state_dict)
    return model


def r3d_18(pretrained=False, progress=True, **kwargs):
    """Construct 18 layer Resnet3D model as in
    https://arxiv.org/abs/1711.11248

    Args:
        pretrained (bool): If True, returns a model pre-trained on Kinetics-400
        progress (bool): If True, displays a progress bar of the download to stderr

    Returns:
        nn.Module: R3D-18 network
    """

    return _video_resnet('r3d_18',
                         pretrained, progress,
                         block=BasicBlock,
                         conv_makers=[Conv3DSimple] * 4,
                         layers=[2, 2, 2, 2],
                         stem=BasicStem, **kwargs)



def mc3_18(pretrained=False, progress=True, **kwargs):
    """Constructor for 18 layer Mixed Convolution network as in
    https://arxiv.org/abs/1711.11248

    Args:
        pretrained (bool): If True, returns a model pre-trained on Kinetics-400
        progress (bool): If True, displays a progress bar of the download to stderr

    Returns:
        nn.Module: MC3 Network definition
    """
    return _video_resnet('mc3_18',
                         pretrained, progress,
                         block=BasicBlock,
                         conv_makers=[Conv3DSimple] + [Conv3DNoTemporal] * 3,
                         layers=[2, 2, 2, 2],
                         stem=BasicStem, **kwargs)



def r2plus1d_18_new_pre(pretrained=False, progress=True, **kwargs):
    """Constructor for the 18 layer deep R(2+1)D network as in
    https://arxiv.org/abs/1711.11248

    Args:
        pretrained (bool): If True, returns a model pre-trained on Kinetics-400
        progress (bool): If True, displays a progress bar of the download to stderr

    Returns:
        nn.Module: R(2+1)D-18 network
    """
    return _video_resnet('r2plus1d_18',
                         pretrained, progress,
                         block=BasicBlock,
                         conv_makers=[Conv2Plus1D] * 4,
                         layers=[2, 2, 2, 2],
                         stem=R2Plus1dStem, **kwargs)
# data=torch.rand(1,3,16,112,112).cuda()
#model=r2plus1d_18_new(pretrained=False, progress=False,num_classes=100).cuda()
# print(model(data).size())