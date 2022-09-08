import numpy as np
import torch
import torch.nn as nn
    
class Block(nn.Module):
    def __init__(self, in_ch, out_ch, bn=False):
        super().__init__()
        self.conv1 = nn.Conv1d(in_ch, out_ch, 3, padding=1) ##
        self.relu  = nn.SiLU()
        self.conv2 = nn.Conv1d(out_ch, out_ch, 3, padding=1) ##
        
        self.conv3 = nn.Conv1d(in_ch, out_ch, 3, padding=1)

    def forward(self, x):
        return self.relu(self.conv2(self.relu(self.conv1(x)))) + self.conv3(x)

class Encoder(nn.Module):
    def __init__(self, chs):
        super().__init__()
        self.enc_blocks = nn.ModuleList([Block(chs[i], chs[i+1]) for i in range(len(chs)-1)])
        self.pool       = nn.MaxPool1d(2)

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
        self.upconvs    = nn.ModuleList([nn.ConvTranspose1d(chs[i], chs[i+1], 2, 2) for i in range(len(chs)-1)])
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

class L63_UnetConvRec_head(nn.Module):
    def __init__(self, enc_chs=(1, 32, 64, 128, 256), dec_chs=(256, 128, 64, 32), num_class=3):
        super().__init__()
        self.encoder     = Encoder(enc_chs)
        self.decoder     = Decoder(dec_chs)
        self.head        = nn.Conv1d(dec_chs[-1], num_class, 1)
        
    def forward(self, x):
        enc_ftrs = self.encoder(x)
        out      = self.decoder(enc_ftrs[::-1][0], enc_ftrs[::-1][1:])

        return self.head(out[-1])
    
class L63_UnetConvRec_dyn(nn.Module):
    def __init__(self, enc_chs=(3, 32, 64, 128, 256), dec_chs=(256, 128, 64, 32), num_class=3):
        super().__init__()
        self.encoder     = Encoder(enc_chs)
        self.decoder     = Decoder(dec_chs)
        self.head        = nn.Conv1d(dec_chs[-1], num_class, 1)
        
    def forward(self, x):
        enc_ftrs = self.encoder(x)
        out      = self.decoder(enc_ftrs[::-1][0], enc_ftrs[::-1][1:])

        return self.head(out[-1]).squeeze(1) + x

class L63_UnetConvRec_sup(nn.Module):
    def __init__(self, enc_chs=(3, 32, 64, 128, 256), dec_chs=(256, 128, 64, 32), num_class=3):
        super().__init__()
        self.encoder     = Encoder(enc_chs)
        self.decoder     = Decoder(dec_chs)
        self.head        = nn.Conv1d(dec_chs[-1], num_class, 1)
        
    def forward(self, x):
        enc_ftrs = self.encoder(x)
        out      = self.decoder(enc_ftrs[::-1][0], enc_ftrs[::-1][1:])

        return self.head(out[-1]).squeeze(1)
    
class L63_DARNN(nn.Module):
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

class L63_DARNN1(nn.Module):
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
        outputs.append(x)
        for i in range(self.step):
            x  = self.model_sup(x)
#             x  = self.model_dyn(x)
            outputs.append(x)

        return outputs