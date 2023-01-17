import numpy
import torch
import torch.nn as nn
from models.TransBTS.Transformer import TransformerModel
from models.TransBTS.PositionalEncoding import FixedPositionalEncoding, LearnedPositionalEncoding
from models.TransBTS.Unet_skipconnection import Unet
from google.colab.patches import cv2_imshow
import matplotlib.pyplot as plt


class TransformerBTS(nn.Module):  # set up transformer model with hyper parameters
    def __init__(
            self,
            img_dim,
            patch_dim,
            num_channels,
            embedding_dim,
            num_heads,
            num_layers,
            hidden_dim,
            dropout_rate=0.0,
            attn_dropout_rate=0.0,
            conv_patch_representation=True,
            positional_encoding_type="learned",
    ):
        super(TransformerBTS, self).__init__()

        assert embedding_dim % num_heads == 0
        assert img_dim % patch_dim == 0

        self.img_dim = img_dim
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.patch_dim = patch_dim
        self.num_channels = num_channels
        self.dropout_rate = dropout_rate
        self.attn_dropout_rate = attn_dropout_rate
        self.conv_patch_representation = conv_patch_representation

        self.num_patches = int((img_dim // patch_dim) ** 3)
        self.seq_length = self.num_patches
        self.flatten_dim = 128 * num_channels

        self.linear_encoding = nn.Linear(self.flatten_dim, self.embedding_dim)
        if positional_encoding_type == "learned":
            self.position_encoding = LearnedPositionalEncoding(
                self.seq_length, self.embedding_dim, self.seq_length
            )
        elif positional_encoding_type == "fixed":
            self.position_encoding = FixedPositionalEncoding(
                self.embedding_dim,
            )

        self.pe_dropout = nn.Dropout(p=self.dropout_rate)

        self.transformer = TransformerModel(
            embedding_dim,
            num_layers,
            num_heads,
            hidden_dim,

            self.dropout_rate,
            self.attn_dropout_rate,
        )
        self.pre_head_ln = nn.LayerNorm(embedding_dim)

        if self.conv_patch_representation:
            self.conv_x = nn.Conv3d(
                128,
                self.embedding_dim,
                kernel_size=3,
                stride=1,
                padding=1
            )

        self.Unet = Unet(in_channels=4, base_channels=16, num_classes=4)  # get values of UNET for encoding/decoding
        self.bn = nn.BatchNorm3d(128)
        self.relu = nn.ReLU(inplace=True)

    def encode(self, x):
        if self.conv_patch_representation:
            # combine embedding with conv patch distribution
            x1_1, x2_1, x3_1, x = self.Unet(x)  # get encoding skip connection masks and final output from model, x
            x = self.bn(x)
            x = self.relu(x)
            x = self.conv_x(x)
            x = x.permute(0, 2, 3, 4, 1).contiguous()
            x = x.view(x.size(0), -1, self.embedding_dim)

        else:
            x = self.Unet(x)
            x = self.bn(x)
            x = self.relu(x)
            x = (
                x.unfold(2, 2, 2)
                    .unfold(3, 2, 2)
                    .unfold(4, 2, 2)
                    .contiguous()
            )
            x = x.view(x.size(0), x.size(1), -1, 8)
            x = x.permute(0, 2, 3, 1).contiguous()
            x = x.view(x.size(0), -1, self.flatten_dim)
            x = self.linear_encoding(x)

        x = self.position_encoding(x)  # get positional encoding data given the smallest image size x
        x = self.pe_dropout(x)  # dropout layer

        # apply transformer
        x, intmd_x = self.transformer(x)  # apply the transformer code after positional encoding
        x = self.pre_head_ln(x)

        return x1_1, x2_1, x3_1, x, intmd_x

    def decode(self, x):
        raise NotImplementedError("Should be implemented in child class!!")

    def forward(self, x, auxillary_output_layers=[1, 2, 3, 4]):

        x1_1, x2_1, x3_1, encoder_output, intmd_encoder_outputs = self.encode(x) # start encoder

        decoder_output = self.decode(
            x1_1, x2_1, x3_1, encoder_output, intmd_encoder_outputs, auxillary_output_layers  # start decoder
        )

        if auxillary_output_layers is not None:
            auxillary_outputs = {}
            for i in auxillary_output_layers:
                val = str(2 * i - 1)
                _key = 'Z' + str(i)
                auxillary_outputs[_key] = intmd_encoder_outputs[val]

            return decoder_output

        return decoder_output

    def _get_padding(self, padding_type, kernel_size):  # generate padding for the images to retain size
        assert padding_type in ['SAME', 'VALID']
        if padding_type == 'SAME':
            _list = [(k - 1) // 2 for k in kernel_size]
            return tuple(_list)
        return tuple(0 for _ in kernel_size)

    def _reshape_output(self, x):  # reshape model output for image processing
        x = x.view(
            x.size(0),
            int(self.img_dim / self.patch_dim),
            int(self.img_dim / self.patch_dim),
            int(self.img_dim / self.patch_dim),
            self.embedding_dim,
        )
        x = x.permute(0, 4, 1, 2, 3).contiguous()

        return x


class BTS(TransformerBTS):  # actual BTS model initialization
    def __init__(
            self,
            img_dim,
            patch_dim,
            num_channels,
            num_classes,
            embedding_dim,
            num_heads,
            num_layers,
            hidden_dim,
            dropout_rate=0.0,
            attn_dropout_rate=0.0,
            conv_patch_representation=True,
            positional_encoding_type="learned",
    ):
        super(BTS, self).__init__(
            img_dim=img_dim,
            patch_dim=patch_dim,
            num_channels=num_channels,
            embedding_dim=embedding_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            hidden_dim=hidden_dim,
            dropout_rate=dropout_rate,
            attn_dropout_rate=attn_dropout_rate,
            conv_patch_representation=conv_patch_representation,
            positional_encoding_type=positional_encoding_type,
        )

        self.num_classes = num_classes

        self.Softmax = nn.Softmax(dim=1)

        # blocks to decode the image blocks into more readable output
        self.Enblock8_1 = EnBlock1(in_channels=self.embedding_dim)
        self.Enblock8_2 = EnBlock2(in_channels=self.embedding_dim // 4)

        self.DeUp4 = DeUp_Cat(in_channels=self.embedding_dim // 4, out_channels=self.embedding_dim // 8)
        self.DeBlock4 = DeBlock(in_channels=self.embedding_dim // 8)

        self.DeUp3 = DeUp_Cat(in_channels=self.embedding_dim // 8, out_channels=self.embedding_dim // 16)
        self.DeBlock3 = DeBlock(in_channels=self.embedding_dim // 16)

        self.DeUp2 = DeUp_Cat(in_channels=self.embedding_dim // 16, out_channels=self.embedding_dim // 32)
        self.DeBlock2 = DeBlock(in_channels=self.embedding_dim // 32)

        self.endconv = nn.Conv3d(self.embedding_dim // 32, 4, kernel_size=1)

    def decode(self, x1_1, x2_1, x3_1, x, intmd_x, intmd_layers=[1, 2, 3, 4]):
        assert intmd_layers is not None, "pass the intermediate layers for MLA"
        encoder_outputs = {}
        all_keys = []
        for i in intmd_layers:  # get the keys from the positional encoder
            val = str(2 * i - 1)
            _key = 'Z' + str(i)
            all_keys.append(_key)
            encoder_outputs[_key] = intmd_x[val]
        all_keys.reverse()

        x8 = encoder_outputs[all_keys[0]]  # use the keys the reshape the output into desired sizes using the position encoder output as a guide
        x8 = self._reshape_output(x8)
        x8 = self.Enblock8_1(x8)
        x8 = self.Enblock8_2(x8)


        # below is repeated for each upsampling step
        y4_1 = self.DeUp4(x8, x3_1)  # (1, 64, 32, 32, 32)  # upsample
        y4 = self.DeBlock4(y4_1)  # remap to fit the encoding

        # below is the code for visualizing the intermediary images.
        # the tensor to be copied can be changed
        # right now it is the encoder positional outputs however x3_1 can be changed to be the y outputs of the model
        new_y4 = x3_1.cpu().detach().numpy()
        new_y4 = new_y4[0][0][:][:][31]
        nrows, ncols = 32, 32
        new_y4 = new_y4.reshape(nrows, ncols)
        inter_img1 = plt.figure()
        inter_img1.add_axes()
        plt.title("intermediary image 32")
        inter_img1 = plt.imshow(
            new_y4, cmap='brg', interpolation='nearest', origin='lower')
        plt.colorbar(inter_img1)
        plt.savefig('inter_img_32.png')
        plt.show(inter_img1)

        y3_1 = self.DeUp3(y4, x2_1)  # (1, 32, 64, 64, 64)
        y3 = self.DeBlock3(y3_1)
        new_y3 = x2_1.cpu().detach().numpy()
        new_y3 = new_y3[0][0][:][:][63]
        nrows, ncols = 64, 64
        new_y3 = new_y3.reshape(nrows, ncols)
        inter_img2 = plt.figure()
        inter_img2.add_axes()
        plt.title("intermediary image 64")
        inter_img2 = plt.imshow(
            new_y3, cmap='brg', interpolation='nearest', origin='lower')
        plt.colorbar(inter_img2)
        plt.savefig('inter_img_64.png')
        plt.show(inter_img2)

        y2_1 = self.DeUp2(y3, x1_1)  # (1, 16, 128, 128, 128)
        y2 = self.DeBlock2(y2_1)
        new_y2 = x1_1.cpu().detach().numpy()
        new_y2 = new_y2[0][0][:][:][127]
        nrows, ncols = 128, 128
        new_y2 = new_y2.reshape(nrows, ncols)
        inter_img3 = plt.figure()
        inter_img3.add_axes()
        plt.title("intermediary image 128")
        inter_img3 = plt.imshow(
            new_y2, cmap='brg', interpolation='nearest', origin='lower')
        plt.colorbar(inter_img3)
        plt.savefig('inter_img_128.png')
        plt.show(inter_img3)

        # do final convolution and use softmax to return the final smoothed image
        y = self.endconv(y2)  # (1, 4, 128, 128, 128)
        y = self.Softmax(y)

        return y


class EnBlock1(nn.Module):  # encoding block to return an initial encoding output
    def __init__(self, in_channels):
        super(EnBlock1, self).__init__()

        self.bn1 = nn.BatchNorm3d(512 // 4)
        self.relu1 = nn.ReLU(inplace=True)
        self.bn2 = nn.BatchNorm3d(512 // 4)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv3d(in_channels, in_channels // 4, kernel_size=3, padding=1)
        self.conv2 = nn.Conv3d(in_channels // 4, in_channels // 4, kernel_size=3, padding=1)

    def forward(self, x):
        x1 = self.conv1(x)
        x1 = self.bn1(x1)
        x1 = self.relu1(x1)
        x1 = self.conv2(x1)
        x1 = self.bn2(x1)
        x1 = self.relu2(x1)

        return x1


class EnBlock2(nn.Module):  # similar to the last encoding block however it is a running sum of the encoding steps
    def __init__(self, in_channels):
        super(EnBlock2, self).__init__()

        self.conv1 = nn.Conv3d(in_channels, in_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm3d(512 // 4)
        self.relu1 = nn.ReLU(inplace=True)
        self.bn2 = nn.BatchNorm3d(512 // 4)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(in_channels, in_channels, kernel_size=3, padding=1)

    def forward(self, x):
        x1 = self.conv1(x)
        x1 = self.bn1(x1)
        x1 = self.relu1(x1)
        x1 = self.conv2(x1)
        x1 = self.bn2(x1)
        x1 = self.relu2(x1)
        x1 = x1 + x

        return x1


class DeUp_Cat(nn.Module):  # decoding concactinate block to combine the outputs of two 3d convolution networks  and step up the number of output channels for each image to generate
    def __init__(self, in_channels, out_channels):
        super(DeUp_Cat, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=1)
        self.conv2 = nn.ConvTranspose3d(out_channels, out_channels, kernel_size=2, stride=2)
        self.conv3 = nn.Conv3d(out_channels * 2, out_channels, kernel_size=1)

    def forward(self, x, prev):
        x1 = self.conv1(x)
        y = self.conv2(x1)
        # y = y + prev
        y = torch.cat((prev, y), dim=1)
        y = self.conv3(y)
        return y


class DeBlock(nn.Module):  # decode convolution block to create a running sum of the decoding output given the encoding mask and model output
    def __init__(self, in_channels):
        super(DeBlock, self).__init__()

        self.bn1 = nn.BatchNorm3d(in_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv3d(in_channels, in_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv3d(in_channels, in_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm3d(in_channels)
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x):
        x1 = self.conv1(x)
        x1 = self.bn1(x1)
        x1 = self.relu1(x1)
        x1 = self.conv2(x1)
        x1 = self.bn2(x1)
        x1 = self.relu2(x1)
        x1 = x1 + x

        return x1


def TransBTS(dataset='brats', _conv_repr=True, _pe_type="learned"):  # initialized the total model for learning and set the hyperparameters
    if dataset.lower() == 'brats':
        img_dim = 128
        num_classes = 4

    num_channels = 4
    patch_dim = 8
    aux_layers = [1, 2, 3, 4]
    model = BTS(
        img_dim,
        patch_dim,
        num_channels,
        num_classes,
        embedding_dim=512,
        num_heads=8,
        num_layers=4,
        hidden_dim=4096,
        dropout_rate=0.1,
        attn_dropout_rate=0.1,
        conv_patch_representation=_conv_repr,
        positional_encoding_type=_pe_type,
    )

    return aux_layers, model


if __name__ == '__main__':  # do model training calulations with gradient calulations off and moved off the gpu to apply the learned behavior to images to cost calulations
    with torch.no_grad():
        import os

        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
        cuda0 = torch.device('cuda:0')
        x = torch.rand((1, 4, 128, 128, 128), device=cuda0)
        _, model = TransBTS(dataset='brats', _conv_repr=True, _pe_type="learned")
        model.cuda()
        y = model(x)
        print(y.shape)
