import copy
import torch.nn as nn
import numpy as np
from abc import abstractmethod

from .util import recursive_clone, skip_sum
from .submodules import ConvLayer, UpsampleConvLayer, TransposedConvLayer, RecurrentConvLayer, ResidualBlock

def copy_states(states):
    """
    LSTM states: [(torch.tensor, torch.tensor), ...]
    GRU states: [torch.tensor, ...]
    """
    if states[0] is None:
        return copy.deepcopy(states)
    return recursive_clone(states)

class BaseModel(nn.Module):
    """
    Base class for all models
    """
    @abstractmethod
    def forward(self, *inputs):
        """
        Forward pass logic

        :return: Model output
        """
        raise NotImplementedError

    def __str__(self):
        """
        Model prints with number of trainable parameters
        """
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return super().__str__() + '\nTrainable parameters: {}'.format(params)

class E2VIDRecurrent(BaseModel):
    """
    Compatible with E2VID_lightweight
    Recurrent, UNet-like architecture where each encoder is followed by a ConvLSTM or ConvGRU.
    """
    def __init__(self, unet_kwargs):
        super().__init__()
        self.num_bins = unet_kwargs['num_bins']  # legacy
        self.num_encoders = unet_kwargs['num_encoders']  # legacy
        self.unetrecurrent = UNetRecurrent(unet_kwargs)

    @property
    def states(self):
        return copy_states(self.unetrecurrent.states)

    @states.setter
    def states(self, states):
        self.unetrecurrent.states = states

    def reset_states(self):
        self.unetrecurrent.states = [None] * self.unetrecurrent.num_encoders

    def forward(self, event_tensor):
        """
        :param event_tensor: N x num_bins x H x W
        :return: output dict with image taking values in [0,1], and
                 displacement within event_tensor.
        """
        output_dict = self.unetrecurrent.forward(event_tensor)
        return output_dict

class UNetRecurrent(nn.Module):
    """
    Compatible with E2VID_lightweight
    Recurrent UNet architecture where every encoder is followed by a recurrent convolutional block,
    such as a ConvLSTM or a ConvGRU.
    Symmetric, skip connections on every encoding layer.
    """
    def __init__(self, unet_kwargs):
        super(UNetRecurrent, self).__init__()
        self.base_num_channels = unet_kwargs['base_num_channels']
        self.num_encoders = unet_kwargs['num_encoders']
        self.num_residual_blocks = unet_kwargs['num_residual_blocks']
        self.num_output_channels = unet_kwargs['num_output_channels']
        self.kernel_size = unet_kwargs['kernel_size']
        self.skip_type = unet_kwargs['skip_type']
        self.norm = unet_kwargs['norm']
        self.num_bins = unet_kwargs['num_bins']
        self.recurrent_block_type = unet_kwargs['recurrent_block_type']
        self.use_upsample_conv = unet_kwargs['use_upsample_conv']
        self.channel_multiplier = unet_kwargs['channel_multiplier']

        self.encoder_input_sizes = [int(self.base_num_channels * pow(self.channel_multiplier, i)) for i in
                                    range(self.num_encoders)]
        self.encoder_output_sizes = [int(self.base_num_channels * pow(self.channel_multiplier, i + 1)) for i in
                                     range(self.num_encoders)]
        self.max_num_channels = self.encoder_output_sizes[-1]

        self.skip_ftn = eval('skip_' + self.skip_type)
        # print('Using skip: {}'.format(self.skip_ftn))
        if self.use_upsample_conv:
            # print('Using UpsampleConvLayer (slow, but no checkerboard artefacts)')
            self.UpsampleLayer = UpsampleConvLayer
        else:
            # print('Using TransposedConvLayer (fast, with checkerboard artefacts)')
            self.UpsampleLayer = TransposedConvLayer
        assert (self.num_output_channels > 0)
        # print('Kernel size ', self.kernel_size)
        # print('Skip type ', self.skip_type)
        # print('norm ', self.norm)

        self.head = ConvLayer(self.num_bins, self.base_num_channels,
                              kernel_size=self.kernel_size, stride=1,
                              padding=self.kernel_size // 2)  # N x C x H x W -> N x 32 x H x W

        self.encoders = nn.ModuleList()
        for input_size, output_size in zip(self.encoder_input_sizes, self.encoder_output_sizes):
            self.encoders.append(RecurrentConvLayer(
                input_size, output_size, kernel_size=self.kernel_size, stride=2,
                padding=self.kernel_size // 2,
                recurrent_block_type=self.recurrent_block_type, norm=self.norm))

        self.build_resblocks()
        self.decoders = self.build_decoders()
        self.states = [None] * self.num_encoders

        # self.conv1 = nn.Conv2d(self.base_num_channels, self.base_num_channels * 2, kernel_size=9, padding=9 // 2)
        # self.conv2 = nn.Conv2d(self.base_num_channels * 2, self.base_num_channels, kernel_size=self.kernel_size, padding=self.kernel_size // 2)
        # self.conv3 = nn.Conv2d(self.base_num_channels, self.num_output_channels, kernel_size=self.kernel_size, padding=self.kernel_size // 2)
        # self.relu = nn.ReLU(inplace=True)

        self.pred = self.build_prediction_layer(self.num_output_channels, self.norm)

    def build_resblocks(self):
        self.resblocks = nn.ModuleList()
        for i in range(self.num_residual_blocks):
            self.resblocks.append(ResidualBlock(self.max_num_channels, self.max_num_channels, norm=self.norm))

    def build_decoders(self):
        decoder_input_sizes = reversed(self.encoder_output_sizes)
        decoder_output_sizes = reversed(self.encoder_input_sizes)
        decoders = nn.ModuleList()
        for input_size, output_size in zip(decoder_input_sizes, decoder_output_sizes):
            decoders.append(self.UpsampleLayer(
                input_size if self.skip_type == 'sum' else 2 * input_size,
                output_size, kernel_size=self.kernel_size,
                padding=self.kernel_size // 2, norm=self.norm))
        # decoders.append(self.UpsampleLayer(
        #     self.base_num_channels if self.skip_type == 'sum' else self.base_num_channels * 2,
        #     self.base_num_channels, kernel_size=self.kernel_size,
        #     padding=self.kernel_size // 2, norm=self.norm))
        return decoders

    def build_prediction_layer(self, num_output_channels, norm=None):
        return ConvLayer(self.base_num_channels if self.skip_type == 'sum' else self.base_num_channels * 2,
                         num_output_channels, 1, activation=None, norm=norm)

    def forward(self, x):
        """
        :param x: N x num_input_channels x H x W
        :return: N x num_output_channels x H x W
        """

        # head
        x = self.head(x) # 2, 5, 180, 240 -> 2, 32, 180, 240
        head = x

        # encoder
        blocks = []
        for i, encoder in enumerate(self.encoders):
            x, state = encoder(x, self.states[i]) # 2, 32, 180, 240 -> 2, 64, 90, 120
            blocks.append(x)
            self.states[i] = state

        # residual blocks
        for resblock in self.resblocks:
            x = resblock(x)

        # decoder
        for i, decoder in enumerate(self.decoders):
            # if i == self.num_encoders:
            #     x = decoder(x)
            # else:
            x = decoder(self.skip_ftn(x, blocks[self.num_encoders - i - 1]))

        # x = self.relu(self.conv1(x))
        # x = self.relu(self.conv2(x))
        # img = self.conv3(x)

        img = self.pred(self.skip_ftn(x, head))

        return {'image': img}

class E2VIDRecurrentUp(BaseModel):
    """
    Compatible with E2VID_lightweight
    Recurrent, UNet-like architecture where each encoder is followed by a ConvLSTM or ConvGRU.
    """
    def __init__(self, unet_kwargs):
        super().__init__()
        self.num_bins = unet_kwargs['num_bins']  # legacy
        self.num_encoders = unet_kwargs['num_encoders']  # legacy
        self.unetrecurrent = UNetRecurrentUp(unet_kwargs)

    @property
    def states(self):
        return copy_states(self.unetrecurrent.states)

    @states.setter
    def states(self, states):
        self.unetrecurrent.states = states

    def reset_states(self):
        self.unetrecurrent.states = [None] * self.unetrecurrent.num_encoders

    def forward(self, event_tensor):
        """
        :param event_tensor: N x num_bins x H x W
        :return: output dict with image taking values in [0,1], and
                 displacement within event_tensor.
        """
        output_dict = self.unetrecurrent.forward(event_tensor)
        return output_dict

class UNetRecurrentUp(nn.Module):
    """
    Compatible with E2VID_lightweight
    Recurrent UNet architecture where every encoder is followed by a recurrent convolutional block,
    such as a ConvLSTM or a ConvGRU.
    Symmetric, skip connections on every encoding layer.
    """
    def __init__(self, unet_kwargs):
        super(UNetRecurrentUp, self).__init__()
        self.base_num_channels = unet_kwargs['base_num_channels']
        self.num_encoders = unet_kwargs['num_encoders']
        self.num_residual_blocks = unet_kwargs['num_residual_blocks']
        self.num_output_channels = unet_kwargs['num_output_channels']
        self.kernel_size = unet_kwargs['kernel_size']
        self.skip_type = unet_kwargs['skip_type']
        self.norm = unet_kwargs['norm']
        self.num_bins = unet_kwargs['num_bins']
        self.recurrent_block_type = unet_kwargs['recurrent_block_type']
        self.use_upsample_conv = unet_kwargs['use_upsample_conv']
        self.channel_multiplier = unet_kwargs['channel_multiplier']

        self.encoder_input_sizes = [int(self.base_num_channels * pow(self.channel_multiplier, i)) for i in
                                    range(self.num_encoders)]
        self.encoder_output_sizes = [int(self.base_num_channels * pow(self.channel_multiplier, i + 1)) for i in
                                     range(self.num_encoders)]
        self.max_num_channels = self.encoder_output_sizes[-1]

        self.skip_ftn = eval('skip_' + self.skip_type)
        print('Using skip: {}'.format(self.skip_ftn))
        if self.use_upsample_conv:
            print('Using UpsampleConvLayer (slow, but no checkerboard artefacts)')
            self.UpsampleLayer = UpsampleConvLayer
        else:
            print('Using TransposedConvLayer (fast, with checkerboard artefacts)')
            self.UpsampleLayer = TransposedConvLayer
        assert (self.num_output_channels > 0)
        print('Kernel size ', self.kernel_size)
        print('Skip type ', self.skip_type)
        print('norm ', self.norm)

        self.head = ConvLayer(self.num_bins, self.base_num_channels,
                              kernel_size=self.kernel_size, stride=1,
                              padding=self.kernel_size // 2)  # N x C x H x W -> N x 32 x H x W

        self.encoders = nn.ModuleList()
        for input_size, output_size in zip(self.encoder_input_sizes, self.encoder_output_sizes):
            self.encoders.append(RecurrentConvLayer(
                input_size, output_size, kernel_size=self.kernel_size, stride=2,
                padding=self.kernel_size // 2,
                recurrent_block_type=self.recurrent_block_type, norm=self.norm))

        self.build_resblocks()
        self.decoders = self.build_decoders()
        self.states = [None] * self.num_encoders

        self.conv1 = nn.Conv2d(self.base_num_channels, self.base_num_channels * 2, kernel_size=9, padding=9 // 2)
        self.conv2 = nn.Conv2d(self.base_num_channels * 2, self.base_num_channels, kernel_size=self.kernel_size, padding=self.kernel_size // 2)
        self.conv3 = nn.Conv2d(self.base_num_channels, self.num_output_channels, kernel_size=self.kernel_size, padding=self.kernel_size // 2)
        self.relu = nn.ReLU(inplace=True)

        self.pred = self.build_prediction_layer(self.num_output_channels, self.norm)

    def build_resblocks(self):
        self.resblocks = nn.ModuleList()
        for i in range(self.num_residual_blocks):
            self.resblocks.append(ResidualBlock(self.max_num_channels, self.max_num_channels, norm=self.norm))

    def build_decoders(self):
        decoder_input_sizes = reversed(self.encoder_output_sizes)
        decoder_output_sizes = reversed(self.encoder_input_sizes)
        decoders = nn.ModuleList()
        for input_size, output_size in zip(decoder_input_sizes, decoder_output_sizes):
            decoders.append(self.UpsampleLayer(
                input_size if self.skip_type == 'sum' else 2 * input_size,
                output_size, kernel_size=self.kernel_size,
                padding=self.kernel_size // 2, norm=self.norm))
        decoders.append(self.UpsampleLayer(
            self.base_num_channels if self.skip_type == 'sum' else self.base_num_channels * 2,
            self.base_num_channels, kernel_size=self.kernel_size,
            padding=self.kernel_size // 2, norm=self.norm))
        return decoders

    def build_prediction_layer(self, num_output_channels, norm=None):
        return ConvLayer(self.base_num_channels if self.skip_type == 'sum' else self.base_num_channels * 2,
                         num_output_channels, 1, activation=None, norm=norm)

    def forward(self, x):
        """
        :param x: N x num_input_channels x H x W
        :return: N x num_output_channels x H x W
        """

        # head
        x = self.head(x) # 2, 5, 180, 240 -> 2, 32, 180, 240
        head = x

        # encoder
        blocks = []
        for i, encoder in enumerate(self.encoders):
            x, state = encoder(x, self.states[i]) # 2, 32, 180, 240 -> 2, 64, 90, 120
            blocks.append(x)
            self.states[i] = state

        # residual blocks
        for resblock in self.resblocks:
            x = resblock(x)

        # decoder
        for i, decoder in enumerate(self.decoders):
            if i == self.num_encoders:
                x = decoder(x)
            else:
                x = decoder(self.skip_ftn(x, blocks[self.num_encoders - i - 1]))

        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        img = self.conv3(x)

        # img = self.pred(self.skip_ftn(x, head))

        return {'image': img}