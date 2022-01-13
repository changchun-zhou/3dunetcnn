import torch
from ..classification.myronenko_th import MyronenkoEncoder
from ..classification.decoder import MirroredDecoder
from ..autoencoder.variational_th import ConvolutionalAutoEncoder
from unet3d.utils.resample import resample_image

class UNetEncoder(MyronenkoEncoder):
    def forward(self, x):
        outputs = list()
        for layer, downsampling in zip(self.layers[:-1], self.downsampling_convolutions):
            x = layer(x)
            outputs.insert(0, x)
            x = downsampling(x)
        x = self.layers[-1](x)
        outputs.insert(0, x)
        return outputs


class UNetDecoder(MirroredDecoder):
    def calculate_layer_widths(self, depth):
        in_width, out_width = super().calculate_layer_widths(depth=depth)
        if depth != len(self.layer_blocks) - 1:
            in_width *= 2
        print("Decoder {}:".format(depth), in_width, out_width)
        return in_width, out_width

    def forward(self, inputs):
        x = inputs[0]
        for i, (pre, up, lay) in enumerate(zip(self.pre_upsampling_blocks, self.upsampling_blocks, self.layers[:-1])):
            x = lay(x)
            x = pre(x)
            x = up(x)
            # print('dimension x: {}, inputs: {}, i: {}'.format(x.size(), inputs[i + 1].size(), i))
            x_dim = inputs[i + 1].size()[4]
            y_dim = inputs[i + 1].size()[3]
            z_dim = inputs[i + 1].size()[2]
            # print('dimension x: {}, inputs: {}, i: {}'.format(x.size(), inputs[i + 1].size(), i))
            # print(x[:, :, 0:z_dim, 0:y_dim, 0:x_dim].size())
            x = torch.cat((x[:, :, 0:z_dim, 0:y_dim, 0:x_dim], inputs[i + 1]), 1)
        x = self.layers[-1](x)
        return x


class UNet(ConvolutionalAutoEncoder):
    def __init__(self, *args, encoder_class=UNetEncoder, decoder_class=UNetDecoder, n_outputs=1, **kwargs):
        super().__init__(*args, encoder_class=encoder_class, decoder_class=decoder_class, n_outputs=n_outputs, **kwargs)
        self.set_final_convolution(n_outputs=n_outputs)


class AutocastUNet(UNet):
    def forward(self, *args, **kwargs):
        from torch.cuda.amp import autocast
        with autocast():
            output = super().forward(*args, **kwargs)
        return output


class AutoImplantUNet(UNet):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x):
        y = super(AutoImplantUNet, self).forward(x)
        return y - x

    def test(self, x):
        return super(AutoImplantUNet, self).forward(x)
