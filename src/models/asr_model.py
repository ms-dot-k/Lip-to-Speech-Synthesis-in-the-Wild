import torch
from torch import nn
from src.models.resnet import ResNetModel
from src.conformer.encoder import ConformerEncoder
from einops import rearrange

class ASR_model(nn.Module):
    def __init__(self, num_layers=8, num_attention_heads=8, num_class=50):
        super().__init__()

        self.audio_front = nn.Sequential(
            nn.Conv2d(1, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
            nn.BatchNorm2d(256),
            nn.ReLU()
            )
        self.embed = nn.Linear(20 * 256, 512)
        self.encoder = ConformerEncoder(encoder_dim=512, num_layers=num_layers, num_attention_heads=num_attention_heads, feed_forward_expansion_factor=4, conv_expansion_factor=2)
        self.classifier = nn.Linear(512, num_class)

    def forward(self, x, vid_len):
        #x:B,1,F,T
        x = self.audio_front(x)
        x = rearrange(x, 'b c f t -> b t (c f)')
        x = self.embed(x)
        mask = self.generate_mask(vid_len, x.size(1)).cuda()
        out_feat = self.encoder(x, mask=mask)
        out = self.classifier(out_feat)
        return out, out_feat

    def generate_mask(self, length, sz):
        masks = []
        for i in range(length.size(0)):
            mask = [0] * length[i]
            mask += [1] * (sz - length[i])
            masks += [torch.tensor(mask)]
        masks = torch.stack(masks, dim=0).bool()
        return masks