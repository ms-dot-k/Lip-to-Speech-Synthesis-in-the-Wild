import torch
from torch import nn
from src.models.resnet import ResNetModel
from src.conformer.encoder import ConformerEncoder
from einops import rearrange

class Visual_front(nn.Module):
    def __init__(self, in_channels=1, conf_layer=8, num_head=8):
        super().__init__()

        self.in_channels = in_channels
        self.frontend = nn.Sequential(
            nn.Conv3d(self.in_channels, 64, kernel_size=(5, 7, 7), stride=(1, 2, 2), padding=(2, 3, 3), bias=False),
            nn.BatchNorm3d(64),
            nn.ReLU(True),
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))
        )

        self.resnet = ResNetModel(
            layers=18,
            output_dim=512,
            pretrained=False,
            large_input=False
        )

        self.dropout = nn.Dropout(0.3)
        self.conformer = Conformer_encoder(conf_layer, num_head)

    def forward(self, x, vid_len):
        #B,C,T,H,W
        T = x.size(2)
        x = self.frontend(x)
        x = self.resnet(x)  # B, T, 512
        x = self.dropout(x)

        mask = self.conformer.generate_mask(vid_len, T).cuda()

        x = self.conformer(x, mask=mask)
        return x

class Conformer_encoder(nn.Module):
    def __init__(self, num_layers=8, num_attention_heads=8):
        super().__init__()

        self.encoder = ConformerEncoder(encoder_dim=512, num_layers=num_layers, num_attention_heads=num_attention_heads, feed_forward_expansion_factor=4, conv_expansion_factor=2)

    def forward(self, x, mask):
        #x:B,T,C
        out = self.encoder(x, mask=mask)
        return out

    def generate_mask(self, length, sz):
        masks = []
        for i in range(length.size(0)):
            mask = [0] * length[i]
            mask += [1] * (sz - length[i])
            masks += [torch.tensor(mask)]
        masks = torch.stack(masks, dim=0).bool()
        return masks

class CTC_classifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.classifier = nn.Linear(512, num_classes)

    def forward(self, x):
        # B, S, 512
        size = x.size()
        x = x.view(-1, size[2]).contiguous()
        x = self.classifier(x)
        return x.view(size[0], size[1], -1)

class Speaker_embed(nn.Module):
    def __init__(self):
        super().__init__()

        self.classifier = nn.Sequential(nn.Conv1d(80, 128, 7, padding=3),
                                        nn.BatchNorm1d(128),
                                        nn.LeakyReLU(0.2),
                                        nn.Conv1d(128, 256, 7, padding=3),
                                        nn.BatchNorm1d(256),
                                        nn.LeakyReLU(0.2),
                                        nn.Conv1d(256, 256, 7, padding=3),
                                        nn.BatchNorm1d(256),
                                        nn.LeakyReLU(0.2)
                                        )

        self.linear = nn.Linear(256, 512)

    def forward(self, x):
        # B, 1, 80, 100
        x = self.classifier(x.squeeze(1))
        x = x.mean(2)
        x = self.linear(x)
        return x    # B, 512

class Mel_classifier(nn.Module):
    def __init__(self):
        super().__init__()

        self.fusion = nn.Sequential(nn.Linear(1024, 512),
                                    nn.ReLU())

        self.classifier = nn.Sequential(nn.Conv1d(512, 256, 7, 1, 3),
                                        nn.ReLU(),
                                        nn.Conv1d(256, 128, 7, 1, 3),
                                        nn.ReLU(),
                                        nn.Conv1d(128, 320, 7, 1, 3),
                                        )

    def forward(self, x, sp):
        sp = sp.unsqueeze(1).repeat(1, x.size(1), 1)
        x = torch.cat([x, sp], 2)
        x = self.fusion(x)  #B, T, 512
        x = x.permute(0, 2, 1).contiguous()     #B, 512, T
        x = self.classifier(x)
        # B, 320, S
        return rearrange(x, 'b (d c f) t -> b d c (t f)', d=1, f=4)

