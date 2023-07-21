import torch
import torch.nn as nn
import torch.nn.functional as F

from woolly.models.common import View

torch.manual_seed(1)

GROUP_SIZE = 2


class GBN(nn.Module):
    def __init__(self, inp, vbs=16, momentum=0.01):
        super().__init__()
        self.bn = nn.BatchNorm1d(inp, momentum=momentum)
        self.vbs = vbs

    def forward(self, x):
        chunk = torch.chunk(x, x.size(0) // self.vbs, 0)
        res = [self.bn(y) for y in chunk]
        return torch.cat(res, 0)


def get_norm_layer(output_size, norm="bn"):
    """This function provides normalization layer based on params

    Args:
        output_size (int): Number of output channel
        norm (str, optional): Parameter to decide which normalization to use,  Allowed values ['bn', 'gn', 'ln']. Defaults to 'bn'.

    Returns:
       nn.Module : Instance of normalization class
    """
    n = nn.BatchNorm2d(output_size)
    if norm == "gn":
        n = nn.GroupNorm(GROUP_SIZE, output_size)
    elif norm == "ln":
        n = nn.GroupNorm(1, output_size)
    elif norm == "gbn":
        n = GBN(output_size)

    return n


class WyConv2d(nn.Module):
    """Creates an instance of 2d convolution based on different params provided


    Args:
        nn (nn.Module): Base Module class
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        padding=1,
        strides=1,
        dilation=1,
        ctype="vanila",
        bias=False,
    ):
        """Init Custom class

        Args:
            in_channels (int): Number of input channels
            out_channels (int): Number of output channels
            kernel_size (int, optional): Kernel size to be used in convolution. Defaults to 3.
            ctype (str, optional): Type of convolution to be used. Allowed values ['vanila', 'depthwise', 'depthwise_seperable'] Defaults to 'vanila'.
            bias (bool, optional): Enable/Disable Bias. Defaults to False.
        """
        super(WyConv2d, self).__init__()
        self.ctype = ctype
        groups = 1
        out = out_channels
        if ctype == "depthwise":
            groups = in_channels
        elif ctype == "depthwise_seperable":
            groups = in_channels
            out = in_channels

        self.conv = nn.Conv2d(
            in_channels,
            out,
            kernel_size=kernel_size,
            stride=strides,
            groups=groups,
            padding=padding,
            dilation=dilation,
            bias=bias,
        )

        if ctype == "depthwise_seperable":
            self.pointwise = nn.Conv2d(
                in_channels, out_channels, kernel_size=1, bias=bias
            )

    def forward(self, x):
        x = self.conv(x)
        if self.ctype == "depthwise_seperable":
            x = self.pointwise(x)
        return x


class WyResidualSE(nn.Module):
    def __init__(
        self,
        input_size,
        output_size,
        padding=1,
        strides=1,
        dilation=1,
        use1x1=False,
        ctype="vanila",
        norm="bn",
        first_block=False,
        last_block=False,
        usedilation=False,
        use_skip=True,
    ):
        super(WyResidualSE, self).__init__()

        self.first_block = first_block

        input_kernels_size = input_size

        if self.first_block:
            output_kernels_size = int(output_size / 2)
        else:
            output_kernels_size = input_kernels_size * 2

        if usedilation:
            print("Pading:", padding, "Sride:", strides, "Dilation:", dilation)
            self.conv1 = WyConv2d(
                input_kernels_size,
                output_kernels_size,
                kernel_size=3,
                padding=1,  # original padding=padding
                strides=1,
                dilation=dilation,
                ctype=ctype,
            )
        else:
            self.conv1 = WyConv2d(
                input_kernels_size,
                output_kernels_size,
                kernel_size=3,
                padding=padding,
                strides=strides,
                dilation=dilation,
                ctype=ctype,
            )
        self.bn1 = get_norm_layer(output_kernels_size, norm=norm)

        input_kernels_size = output_kernels_size
        output_kernels_size = input_kernels_size * 2

        self.conv2 = WyConv2d(
            input_kernels_size,
            output_kernels_size,
            kernel_size=3,
            padding=1,
            strides=1,
            ctype=ctype,
        )
        self.bn2 = get_norm_layer(output_kernels_size, norm=norm)

        # input_kernels_size = output_kernels_size
        # output_kernels_size = input_kernels_size*2

        # self.conv3 = WyConv2d(
        #     input_kernels_size, output_kernels_size, kernel_size=3, padding=1, strides=1, ctype=ctype
        # )
        # self.bn3 = get_norm_layer(output_kernels_size, norm=norm)

        input_kernels_size = output_kernels_size
        output_kernels_size = output_size

        self.squeeze = WyConv2d(
            input_kernels_size,
            output_kernels_size,
            kernel_size=1,
            padding=0,
            strides=1,
            ctype=ctype,
        )

        self.pointwise = nn.Sequential()
        self.use_skip = use_skip

        if use1x1 and use_skip and input_size != output_size:  # and strides != 1:
            self.pointwise = WyConv2d(
                input_size, output_size, kernel_size=1, padding=0, strides=strides
            )

    def forward(self, x):
        y = F.relu(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        # y = F.relu(self.bn3(self.conv3(y)))
        y = self.squeeze(y)
        if self.pointwise:
            x = self.pointwise(x)

        # if not self.first_block and self.use_skip:
        #     y += x

        if self.use_skip:
            y += x

        return F.relu(y)


class WyResidual(nn.Module):
    def __init__(
        self,
        input_size,
        output_size,
        padding=1,
        strides=1,
        dilation=1,
        use1x1=False,
        ctype="vanila",
        norm="bn",
        first_block=False,
        last_block=False,
        usedilation=False,
        use_skip=True,
    ):
        super(WyResidual, self).__init__()

        self.first_block = first_block

        if usedilation:
            print("Pading:", padding, "Sride:", strides, "Dilation:", dilation)
            self.conv1 = WyConv2d(
                input_size,
                output_size,
                kernel_size=3,
                padding=1,  # original padding=padding
                strides=1,
                dilation=dilation,
                ctype=ctype,
            )
        else:
            self.conv1 = WyConv2d(
                input_size,
                output_size,
                kernel_size=3,
                padding=padding,
                strides=strides,
                dilation=dilation,
                ctype=ctype,
            )
        self.bn1 = get_norm_layer(output_size, norm=norm)

        self.conv2 = WyConv2d(
            output_size, output_size, kernel_size=3, padding=1, strides=1, ctype=ctype
        )
        self.bn2 = get_norm_layer(output_size, norm=norm)

        # self.conv3 = WyConv2d(
        #     output_size, output_size, kernel_size=3, padding=1, strides=1, ctype=ctype
        # )
        # self.bn3 = get_norm_layer(output_size, norm=norm)

        self.pointwise = nn.Sequential()
        self.use_skip = use_skip

        if use1x1 and use_skip and input_size != output_size:  # and strides != 1:
            self.pointwise = nn.Sequential(
                WyConv2d(
                    input_size, output_size, kernel_size=1, padding=0, strides=strides
                ),
                get_norm_layer(output_size, norm=norm),
            )

    def forward(self, x):
        y = F.relu(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        if self.pointwise:
            x = self.pointwise(x)

        # if not self.first_block and self.use_skip:
        #     y += x

        if self.use_skip:
            y += x

        return F.relu(y)


class WyBlock(nn.Module):
    def __init__(
        self,
        input_size,
        output_size,
        repetations=2,
        ctype="vanila",
        norm="bn",
        padding=1,
        strides=2,
        dilation=1,
        use1x1=False,
        usepool=False,
        usedilation=False,
        use_skip=True,
        first_block=False,
        last_block=False,
    ):
        """Initialize Block

        Args:
            input_size (int): Input Channel Size
            output_size (int): Output Channel Size
            padding (int, optional): Padding to be used for convolution layer. Defaults to 1.
            norm (str, optional): Type of normalization to be used. Allowed values ['bn', 'gn', 'ln']. Defaults to 'bn'.
            usepool (bool, optional): Enable/Disable Maxpolling. Defaults to True.
        """
        super(WyBlock, self).__init__()
        self.usepool = usepool
        self.last_block = last_block
        self.wyresudals = []
        for r in range(repetations):
            if r == 0:
                if usedilation:
                    self.wyresudals.append(
                        WyResidual(
                            input_size,
                            output_size,
                            padding=0,
                            strides=strides,
                            dilation=dilation,
                            use1x1=use1x1,
                            ctype=ctype,
                            norm=norm,
                            usedilation=usedilation,
                            use_skip=use_skip,
                            first_block=first_block,
                            last_block=last_block,
                        )
                    )
                else:
                    self.wyresudals.append(
                        WyResidual(
                            input_size,
                            output_size,
                            padding=padding,
                            strides=strides,
                            dilation=dilation,
                            use1x1=use1x1,
                            ctype=ctype,
                            norm=norm,
                            usedilation=usedilation,
                            use_skip=use_skip,
                            first_block=first_block,
                            last_block=last_block,
                        )
                    )
            else:
                self.wyresudals.append(
                    WyResidual(
                        output_size,
                        output_size,
                        padding=padding,
                        use1x1=use1x1,
                        ctype=ctype,
                        norm=norm,
                        usedilation=usedilation,
                        use_skip=use_skip,
                        first_block=first_block,
                        last_block=last_block,
                    )
                )

        self.conv = nn.Sequential(*self.wyresudals)

        if usepool and not last_block:
            self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        """
        Args:
            x (tensor): Input tensor to this block

        Returns:
            tensor: Return processed tensor
        """
        x = self.conv(x)

        if self.usepool and not self.last_block:
            x = self.pool(x)

        return x


class WyCifar10Net(nn.Module):
    """Network Class

    Args:
        nn (nn.Module): Instance of pytorch Module
    """

    def __init__(
        self,
        image,
        input_size=3,
        classes=10,
        base_channels=4,
        layers=3,
        drop_ratio=0.01,
        ctype="vanila",
        norm="bn",
        use1x1=False,
        usedilation=False,
        use_skip=True,
        blocks_count=2,
        usepool=False,
    ):
        """Initialize Network

        Args:
            base_channels (int, optional): Number of base channels to start with. Defaults to 4.
            layers (int, optional): Number of Layers in each block. Defaults to 3.
            drop (float, optional): Dropout value. Defaults to 0.01.
            norm (str, optional): Normalization type. Defaults to 'bn'.
        """
        # Variables
        self.input_size = input_size
        self.classes = classes
        self.base_channels = base_channels
        self.layers = layers
        self.drop_ratio = drop_ratio
        self.ctype = ctype
        self.norm = norm
        self.use1x1 = use1x1
        self.use_skip = use_skip
        self.height, self.width = image
        self.dilation = (1, 1)
        self.blocks_count = blocks_count
        self.usepool = usepool

        self.blocks = []

        self.output_height, self.output_width = image

        super(WyCifar10Net, self).__init__()

        # Base Block
        # self.blocks.append(
        #     WyResidual(input_size, self.base_channels * 2, first_block=True)
        # )
        self.blocks.append(
            WyBlock(
                input_size,
                self.base_channels * 2,
                repetations=self.layers,
                ctype=self.ctype,
                norm=self.norm,
                padding=1,
                dilation=self.dilation,
                use1x1=self.use1x1,
                ## Make sure to not change input size in the very first block
                strides=1,
                ## Make this to use pool else make usepool=False
                usepool=self.usepool,  # False,
                # ###########################################################
                usedilation=usedilation,
                use_skip=self.use_skip,
                first_block=True,
            )
        )

        k = 3  # Original Size of kernel
        k = k + (k - 1) * (self.dilation[0] - 1)  # Kernel size after dilation

        print("KernelSize:", k)

        # Size of output block after using dilation as the first step other than in the first block where dilation is 1
        self.output_height = (self.output_height + 1 + 1 - k) / (1) + 1
        self.output_width = (self.output_width + 1 + 1 - k) / (1) + 1

        print("OutputHeight:", self.output_height, "OutputWidth:", self.output_width)

        for p in range(self.blocks_count):
            if p + 1 == self.blocks_count:
                self.blocks.append(self._make_block(usedilation, last_block=True))
            else:
                self.blocks.append(self._make_block(usedilation))

        # Combine Feature Layer
        self.feature = nn.Sequential(*self.blocks)

        # Output Block
        # self.gap = nn.AdaptiveAvgPool2d(1)
        # self.flat = nn.Conv2d(self.base_channels*2, self.classes, 1)
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            # View(self.base_channels * 2),
            # nn.Linear(self.base_channels * 2, self.classes)
            nn.Conv2d(self.base_channels * 2, self.classes, 1),
        )

    def _make_block(self, usedilation, last_block=False):
        # Desired Image Size after block
        desired_height, desired_width = int(self.output_height / 2), int(
            self.output_width / 2
        )
        kernel_size_h, kernel_size_w = (3, 3)
        p_l, p_r, p_t, p_b = (1, 1, 1, 1)
        s = 1

        # desired_kernel_size_h, desired_kernel_size_w = (kernel_size_h+(kernel_size_h-1)*(self.dilation[0]-1), kernel_size_w+(kernel_size_w-1)*(self.dilation[1]-1))
        # desired_width = ((self.output_width + p_l + p_r - (kernel_size_w+(kernel_size_w-1)*(self.dilation[1]-1))) / (s)) + 1
        # (desired_width - 1) = (self.output_width + p_l + p_r - (kernel_size_w+(kernel_size_w-1)*(self.dilation[1]-1))) / (s)
        # (desired_width - 1) * s = self.output_width + p_l + p_r - (kernel_size_w + (kernel_size_w-1) * (self.dilation[1]-1))
        # ((desired_width - 1) * s) - self.output_width - p_l - p_r = - kernel_size_w - (kernel_size_w-1) * (self.dilation[1]-1)
        # ((desired_width - 1) * s) - self.output_width - p_l - p_r + kernel_size_w = - ((kernel_size_w-1) * (x-1))
        # ((desired_width - 1) * s) - self.output_width - p_l - p_r + kernel_size_w = kernel_size_w - x * kernel_size_w + x - 1
        # ((desired_width - 1) * s) - self.output_width - p_l - p_r + 1 = - x * kernel_size_w + x
        # ((desired_width - 1) * s) - self.output_width - p_l - p_r + 1 = x * (1-kernel_size_w)

        if usedilation:
            self.dilation = (
                int(
                    (((desired_width - 1) * s) - self.output_width - p_l - p_r + 1)
                    / (1 - kernel_size_w)
                ),
                int(
                    (((desired_height - 1) * s) - self.output_height - p_t - p_b + 1)
                    / (1 - kernel_size_h)
                ),
            )

        # if usedilation:
        #     self.dilation = (max(int(self.output_height / 4), 1), max(int(self.output_width / 4), 1))
        self.base_channels = self.base_channels * 2
        print("Dilation", self.dilation)
        block = WyBlock(
            self.base_channels,
            self.base_channels * 2,
            repetations=self.layers,
            ctype=self.ctype,
            norm=self.norm,
            padding=1,
            dilation=self.dilation,
            use1x1=self.use1x1,
            ## Make this to use pool else remove stride and make usepool=False
            strides=1 if self.usepool else 2,
            usepool=self.usepool,  # False,
            #######################
            usedilation=usedilation,
            use_skip=self.use_skip,
            last_block=last_block,
        )
        self.height, self.width = self.height / 2, self.width / 2
        print("Height:", self.height, "Width:", self.width)

        # Divide the height by 2 as we are using strides / max polling for height change
        # Effective Kernel size calculation when using dilation
        k = 3
        k = k + (k - 1) * (self.dilation[0] - 1)

        print("KernelSize:", k)

        self.output_height = (self.output_height + 1 + 1 - k) / (1) + 1
        self.output_width = (self.output_width + 1 + 1 - k) / (1) + 1

        print("OutputHeight:", self.output_height, "OutputWidth:", self.output_width)

        return block

    def forward(self, x, use_softmax=False, dropout=False):
        """Convolution function

        Args:
            x (tensor): Input image tensor
            dropout (bool, optional): Enable/Disable Dropout. Defaults to True.

        Returns:
            tensor: tensor of logits
        """

        # Feature Layer
        x = self.feature(x)

        # Classifier Layer
        x = self.classifier(x)

        # Reshape
        x = x.view(-1, self.classes)

        # Output Layer
        return F.log_softmax(x, dim=1) if use_softmax else x
