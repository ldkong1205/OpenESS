import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as f

from models.submodules import InterpolationLayer


class SemSegE2VID(nn.Module):

    def __init__(
        self,
        input_c, output_c, skip_connect=False, skip_type='sum', input_index_map=False,
        text_embeddings_path='', if_linear_probing=False,
    ):
        super(SemSegE2VID, self).__init__()
        self.skip_connect = skip_connect  # True
        self.skip_type = skip_type  # 'concat'
        self.apply_skip_connection = skip_sum if self.skip_type == 'sum' else skip_concat
        tch = input_c  # 256
        self.index_coords = None
        self.input_index_map = input_index_map  # False

        self.text_embeddings_path = text_embeddings_path
        text_categories = output_c
        if self.text_embeddings_path is None:
            self.text_embeddings = nn.Parameter(torch.zeros(text_categories, 512))
            nn.init.normal_(self.text_embeddings, mean=0.0, std=0.01)
        else:
            self.register_buffer('text_embeddings', torch.randn(text_categories, 512))
            loaded = torch.load(self.text_embeddings_path, map_location='cuda')
            self.text_embeddings[:, :] = loaded[:, :]

        if self.skip_connect:
            decoder_list_1 = []
            for i in range(0, 5):  # 3, 5
                decoder_list_1 += [INSResBlock(tch, tch)]
            decoder_list_1 += [
                ReLUINSConv2d(tch, tch // 2, kernel_size=3, stride=1, padding=1)
            ]

            self.decoder_scale_1 = torch.nn.Sequential(*decoder_list_1)

            self.decoder_scale_2 = nn.Sequential(
                ReLUINSConv2d(tch, tch // 2, kernel_size=3, stride=1, padding=1),
                ReLUINSConv2d(tch // 2, tch // 4, kernel_size=3, stride=1, padding=1),
            )
            tch = tch // 2

            self.decoder_scale_3 = nn.Sequential(
                ReLUINSConv2d(tch, tch // 2, kernel_size=3, stride=1, padding=1),
                ReLUINSConv2d(tch // 2, tch // 2, kernel_size=3, stride=1, padding=1),
            )
            tch = tch // 2

            self.decoder_scale_4 = nn.Sequential(
                ReLUINSConv2d(tch, tch // 2, kernel_size=3, stride=1, padding=1),
            )
            tch = tch // 2

            self.decoder_scale_5 = nn.Sequential(
                torch.nn.Conv2d(tch, output_c, kernel_size=1, stride=1, padding=0),
            )

            self.decoder_ch256 = nn.Sequential(
                torch.nn.Conv2d(tch, 256, kernel_size=1, stride=1, padding=0),
            )

            self.decoder_ch512 = nn.Sequential(
                torch.nn.Conv2d(256, 512, kernel_size=1, stride=1, padding=0),
            )

        else:
            if self.input_index_map:
                tch += 2
            # Instance Norm
            decoder_list_1 = []
            for i in range(0, 3):
                decoder_list_1 += [INSResBlock(tch, tch)]

            self.decoder_scale_1 = torch.nn.Sequential(*decoder_list_1)
            tch = tch

            if self.input_index_map:
                self.decoder_scale_2 = nn.Sequential(
                    InterpolationLayer(scale_factor=2, mode='nearest'),
                    ReLUINSConv2d(tch, (tch - 2) // 2, kernel_size=3, stride=1, padding=1),
                )
                tch = (tch - 2) // 2
            else:
                self.decoder_scale_2 = nn.Sequential(
                    InterpolationLayer(scale_factor=2, mode='nearest'),
                    ReLUINSConv2d(tch, tch // 2, kernel_size=3, stride=1, padding=1),
                )
                tch = tch // 2
            
            self.decoder_scale_3 = nn.Sequential(
                InterpolationLayer(scale_factor=2, mode='nearest'),
                ReLUINSConv2d(tch, tch // 2, kernel_size=3, stride=1, padding=1),
            )
            tch = tch // 2
            tch = tch

            self.decoder_scale_4 = nn.Sequential(
                InterpolationLayer(scale_factor=2, mode='nearest'),
                ReLUINSConv2d(tch, tch // 2, kernel_size=3, stride=1, padding=1),
            )
            tch = tch // 2
            self.decoder_scale_5 = nn.Sequential(
                torch.nn.Conv2d(tch, output_c, kernel_size=1, stride=1, padding=0),
            )

        self.if_linear_probing = if_linear_probing
        if self.if_linear_probing:
            for param in self.decoder_scale_1.parameters():
                param.requires_grad = False

            for param in self.decoder_scale_2.parameters():
                param.requires_grad = False

            for param in self.decoder_scale_3.parameters():
                param.requires_grad = False

            for param in self.decoder_scale_4.parameters():
                param.requires_grad = False

            for param in self.decoder_ch256.parameters():
                param.requires_grad = False

            for param in self.decoder_ch512.parameters():
                param.requires_grad = False

            self.linear_probe = nn.Conv2d(text_categories, text_categories, 1)

    def update_skip_dict(self, skips, x, sz_in):
        rem, scale = sz_in % x.shape[3], sz_in // x.shape[3]
        assert rem == 0
        skips[scale] = x

    def forward(self, input_dict):
        sz_in = input_dict[1].shape[3]  # 640 / 352

        x = input_dict[8]  # [bs, 256, 55, 80] / [bs, 256, 25, 44]
        out = {8: x}

        if self.skip_connect:
            x = self.decoder_scale_1(x)  # [bs, 128, 55, 80] / [bs, 128, 25, 44]
            x = f.interpolate(x, scale_factor=2, mode='nearest')  # [bs, 128, 110, 160] / [bs, 128, 50, 88]
            x = self.apply_skip_connection(x, input_dict[4])  # [bs, 256, 110, 160] / [bs, 256, 50, 88]

            x = self.decoder_scale_2(x)  # [bs, 64, 110, 160] / [bs, 64, 50, 88]
            self.update_skip_dict(out, x, sz_in)
            x = f.interpolate(x, scale_factor=2, mode='nearest')  # [bs, 64, 220, 320] / [bs, 64, 100, 176]
            x = self.apply_skip_connection(x, input_dict[2])  # [bs, 128, 220, 320] / [bs, 128, 100, 176]

            x = self.decoder_scale_3(x)  # [bs, 64, 220, 320] / [bs, 64, 100, 176]
            self.update_skip_dict(out, x, sz_in)
            x = f.interpolate(x, scale_factor=2, mode='nearest')  # [bs, 64, 440, 640] / [bs, 64, 200, 352]

            x = self.decoder_scale_4(x)  # [bs, 32, 440, 640] / [bs, 32, 200, 352]

            # clip
            x_ch256 = self.decoder_ch256(x)  # [bs, 256, 440, 640] / [bs, 256, 200, 352]
            x = self.decoder_ch512(x_ch256)  # [bs, 512, 440, 640] / [bs, 512, 200, 352]
            x = f.conv2d(x, self.text_embeddings[:, :, None, None])  # logits

            # x = self.decoder_scale_5(x)  # [bs, cls, 440, 640]

            if self.if_linear_probing:
                x = self.linear_probe(x)

            self.update_skip_dict(out, x, sz_in)
        else:
            if self.input_index_map:
                if self.index_coords is None or self.index_coords.size(2) != x.size(2):
                    x_coords = torch.arange(
                        x.size(2), device=x.device, dtype=torch.float,
                    )
                    y_coords = torch.arange(
                        x.size(3), device=x.device, dtype=torch.float,
                    )
                    self.index_coords = torch.stack(
                        torch.meshgrid([x_coords, y_coords]), dim=0,
                    )
                    self.index_coords = self.index_coords[None, :, :, :].repeat([x.size(0), 1, 1, 1])
                
                x = torch.cat([x, self.index_coords], dim=1)

            x = self.decoder_scale_1(x)
            x = self.decoder_scale_2(x)
            self.update_skip_dict(out, x, sz_in)
            x = self.decoder_scale_3(x)
            self.update_skip_dict(out, x, sz_in)
            x = self.decoder_scale_4(x)
            x = self.decoder_scale_5(x)
            self.update_skip_dict(out, x, sz_in)
        
        return out, x_ch256


class StyleEncoderE2VID(nn.Module):
    def __init__(self, input_dim, skip_connect=False):
        super(StyleEncoderE2VID, self).__init__()
        conv_list = []
        self.skip_connect = skip_connect

        conv_list += [
            nn.Conv2d(input_dim, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        ]
        conv_list += list(models.resnet18(pretrained=True).children())[1:3]
        conv_list += list(models.resnet18(pretrained=True).children())[4:5]
        
        self.encoder_scale_1 = nn.Sequential(*conv_list)
        self.encoder_scale_2 = list(models.resnet18(pretrained=True).children())[5]
        self.encoder_scale_3 = list(models.resnet18(pretrained=True).children())[6]

    def update_skip_dict(self, skips, x, sz_in):
        rem, scale = sz_in % x.shape[3], sz_in // x.shape[3]
        assert rem == 0
        skips[scale] = x

    def forward(self, x):
        out = {1: x}
        sz_in = x.shape[3]

        if self.skip_connect:
            x = self.encoder_scale_1(x)
            self.update_skip_dict(out, x, sz_in)
            x = self.encoder_scale_2(x)
            self.update_skip_dict(out, x, sz_in)
            x = self.encoder_scale_3(x)
            self.update_skip_dict(out, x, sz_in)
        else:
            x = self.encoder_scale_1(x)
            x = self.encoder_scale_2(x)
            x = self.encoder_scale_3(x)
            self.update_skip_dict(out, x, sz_in)

        return out


####################################################################
# -------------------------- Basic Blocks --------------------------
####################################################################

def gaussian_weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 and classname.find('Conv') == 0:
        m.weight.data.normal_(0.0, 0.02)


class ReLUINSConv2d(nn.Module):
    def __init__(self, n_in, n_out, kernel_size, stride, padding=0):
        super(ReLUINSConv2d, self).__init__()
        model = []
        model += [nn.Conv2d(n_in, n_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=True)]
        model += [nn.InstanceNorm2d(n_out, affine=False)]
        model += [nn.ReLU(inplace=True)]
        self.model = nn.Sequential(*model)
        self.model.apply(gaussian_weights_init)

    def forward(self, x):
        return self.model(x)


class INSResBlock(nn.Module):
    def conv3x3(self, inplanes, out_planes, stride=1):
        return [nn.Conv2d(inplanes, out_planes, kernel_size=3, stride=stride, padding=1)]

    def __init__(self, inplanes, planes, stride=1, dropout=0.0):
        super(INSResBlock, self).__init__()
        model = []
        model += self.conv3x3(inplanes, planes, stride)
        model += [nn.InstanceNorm2d(planes)]
        model += [nn.ReLU(inplace=True)]
        model += self.conv3x3(planes, planes)
        model += [nn.InstanceNorm2d(planes)]

        if dropout > 0:
            model += [nn.Dropout(p=dropout)]
        
        self.model = nn.Sequential(*model)
        self.model.apply(gaussian_weights_init)

    def forward(self, x):
        residual = x
        out = self.model(x)
        out += residual
        return out


def skip_concat(x1, x2):
    return torch.cat([x1, x2], dim=1)


def skip_sum(x1, x2):
    return x1 + x2



