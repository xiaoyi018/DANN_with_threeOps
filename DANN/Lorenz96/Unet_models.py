import numpy as np
import torch
import torch.nn as nn

class Conv2dReccurent(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(Conv2dReccurent, self).__init__()
        self.in_channels  = in_channels
        self.out_channels = out_channels
        self.kernel_size  = kernel_size
        self.stride       = stride
        self.padding      = padding
        self.conv         = nn.Conv2d(in_channels=self.in_channels, out_channels=self.out_channels, 
                                      kernel_size=self.kernel_size, stride=self.stride, padding=self.padding)

    def forward(self, x):
        if isinstance(self.kernel_size, tuple):
            margin = self.kernel_size[0] // 2
        else:
            margin = self.kernel_size // 2
        origin_size = x.shape[2] // self.stride
        if margin != 0:
            x = torch.cat([x[:, :, -margin:, :], x, x[:, :, :margin, :]], axis=2)
            x = self.conv(x)
            new_size = x.shape[2]
            x = x[:,:,(new_size-origin_size)//2:(new_size+origin_size)//2,:]
        else:
            x = self.conv(x)
        return x

class ConvTranspose2dReccurent(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, output_padding=0):
        super(ConvTranspose2dReccurent, self).__init__()
        self.in_channels    = in_channels
        self.out_channels   = out_channels
        self.kernel_size    = kernel_size
        self.stride         = stride
        self.padding        = padding
        self.output_padding = output_padding
        self.conv           = nn.ConvTranspose2d(in_channels=self.in_channels, out_channels=self.out_channels,
                                                 kernel_size=self.kernel_size, stride=self.stride, padding=self.padding, output_padding=self.output_padding)

    def forward(self, x):
        if isinstance(self.kernel_size, tuple):
            margin = self.kernel_size[0] // 2
        else:
            margin = self.kernel_size // 2
        origin_size = x.shape[2] * self.stride
        if margin != 0:
            x = torch.cat([x[:, :, -margin:, :], x, x[:, :, :margin, :]], axis=2)
            x = self.conv(x)
            new_size = x.shape[2]
            x = x[:,:,(new_size-origin_size)//2:(new_size+origin_size)//2,:]
        else:
            x = self.conv(x)
        return x
    
class Block(nn.Module):
    def __init__(self, in_ch, out_ch, bn=False):
        super().__init__()
        self.conv1 = Conv2dReccurent(in_ch, out_ch, 3, padding=1) ##
        self.relu  = nn.SiLU()
        self.conv2 = Conv2dReccurent(out_ch, out_ch, 3, padding=1) ##
        self.bn    = nn.BatchNorm2d(out_ch)
        
        self.conv3 = Conv2dReccurent(in_ch, out_ch, 3, padding=1)

    def forward(self, x):
        return self.relu(self.conv2(self.relu(self.conv1(x)))) + self.conv3(x)

class Encoder(nn.Module):
    def __init__(self, chs):
        super().__init__()
        self.enc_blocks = nn.ModuleList([Block(chs[i], chs[i+1]) for i in range(len(chs)-1)])
        self.pool       = nn.MaxPool2d((1, 2))

    def forward(self, x):
        ftrs = []
        for i, block in enumerate(self.enc_blocks):
            x = block(x)
            ftrs.append(x)
            x = self.pool(x)
        return ftrs

class Decoder(nn.Module):
    def __init__(self, chs):
        super().__init__()
        self.chs        = chs
        self.upconvs    = nn.ModuleList([ConvTranspose2dReccurent(chs[i], chs[i+1], (1, 2), (1, 2)) for i in range(len(chs)-1)])
        ##
        self.dec_blocks = nn.ModuleList([Block(chs[i], chs[i+1]) for i in range(len(chs)-1)])

    def forward(self, x, encoder_features):
        decoder_features = [x]
        for i in range(len(self.chs)-1):
            x        = self.upconvs[i](x)
            enc_ftrs = encoder_features[i]
            x        = torch.cat([x, enc_ftrs], dim=1)
            x        = self.dec_blocks[i](x)
            decoder_features.append(x)
        return decoder_features

class L96_UnetConvRec_head(nn.Module):
    def __init__(self, enc_chs=(1, 32, 64, 128), dec_chs=(128, 64, 32), num_class=1):
        super().__init__()
        self.encoder     = Encoder(enc_chs)
        self.decoder     = Decoder(dec_chs)
        self.conv1       = Block(1, 32)
        self.conv2       = Block(128+32, 128)
        self.head        = Conv2dReccurent(dec_chs[-1], num_class, 1)
        
    def forward(self, x):
        x        = x.unsqueeze(1)
        enc_ftrs = self.encoder(x)
        y        = self.conv1(x[:,:,:,::4])
        z        = self.conv2(torch.cat([y, enc_ftrs[::-1][0]], dim=1))
        out      = self.decoder(z, enc_ftrs[::-1][1:])

        return self.head(out[-1]).squeeze(1)
    
class L96_UnetConvRec_dyn(nn.Module):
    def __init__(self, enc_chs=(1, 32, 64, 128), dec_chs=(128, 64, 32), num_class=1):
        super().__init__()
        self.encoder     = Encoder(enc_chs)
        self.decoder     = Decoder(dec_chs)
        self.head        = Conv2dReccurent(dec_chs[-1], num_class, 1)
        
    def forward(self, xinp):
        x        = xinp.unsqueeze(1)
        enc_ftrs = self.encoder(x)
        out      = self.decoder(enc_ftrs[::-1][0], enc_ftrs[::-1][1:])

        return self.head(out[-1]).squeeze(1) + xinp

class L96_UnetConvRec_sup(nn.Module):
    def __init__(self, enc_chs=(1, 32, 64, 128), dec_chs=(128, 64, 32), num_class=1):
        super().__init__()
        self.encoder     = Encoder(enc_chs)
        self.decoder     = Decoder(dec_chs)
        self.head        = Conv2dReccurent(dec_chs[-1], num_class, 1)
        
    def forward(self, xinp):
        x        = xinp.unsqueeze(1)
        enc_ftrs = self.encoder(x)
        out      = self.decoder(enc_ftrs[::-1][0], enc_ftrs[::-1][1:])

        return self.head(out[-1]).squeeze(1)
    
class L96_DARNN(nn.Module):
    def __init__(self, model_head, model_dyn, model_sup=None, step=0):
        super().__init__()
        self.step        = step
        self.model_head  = model_head
        self.model_dyn   = model_dyn
        self.model_sup   = model_sup
        
    def forward(self, xinp, mask):
        outputs = []
        x   = self.model_head(xinp * mask)
        x   = self.model_dyn(x)
        outputs.append(x)
        for i in range(self.step):
            x  = self.model_sup(x)
            x  = self.model_dyn(x)
            outputs.append(x)

        return outputs

class L96_DARNN1(nn.Module):
    def __init__(self, model_head, model_dyn, model_sup=None, step=0):
        super().__init__()
        self.step        = step
        self.model_head  = model_head
        self.model_dyn   = model_dyn
        self.model_sup   = model_sup
        
    def forward(self, xinp, mask):
        outputs = []
        x   = self.model_head(xinp * mask)
#         x   = self.model_dyn(x)
#         outputs.append(x)
        for i in range(self.step):
            x  = self.model_sup(x)
            x  = self.model_dyn(x)
            outputs.append(x)

        return outputs