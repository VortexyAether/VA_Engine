import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange
import math

class TimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

class SelfAttention(nn.Module):
    def __init__(self, in_dim):
        super(SelfAttention, self).__init__()
        self.chanel_in = in_dim
        self.query_conv = nn.Conv2d(in_dim, in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_dim, in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_dim, in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        B, C, W, H = x.size()
        proj_query = self.query_conv(x).view(B, -1, W * H).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(B, -1, W * H)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = self.value_conv(x).view(B, -1, W * H)
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(B, C, W, H)
        out = self.gamma * out + x
        return out

class ResNetBlock(nn.Module):
    def __init__(self, channels_in, channels_out):
        super().__init__()
        self.conv1 = nn.Conv2d(channels_in, channels_out, kernel_size=3, padding=1)
        self.norm1 = nn.GroupNorm(32, channels_out)
        self.silu = nn.SiLU()

#        self.time_mlp = nn.Sequential(
#            nn.SiLU(),
#            nn.Linear(time_emb_dim, channels_out)
#        )

        self.conv2 = nn.Conv2d(channels_out, channels_out, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(32, channels_out)

        self.residual_connection = nn.Conv2d(channels_in, channels_out, kernel_size=1)

    def forward(self, x):
        residual = self.residual_connection(x)

        x = self.conv1(x)
        x = self.norm1(x)

        #time_emb = self.time_mlp(t)
        #x += time_emb.unsqueeze(-1).unsqueeze(-1)


        x = self.silu(x)
        x = self.conv2(x)
        x = self.norm2(x)
        x += residual

        return x


class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, init_features=64):
        super(UNet, self).__init__()

#        self.time_emb_dim = 64 
#        self.time_embedding = TimeEmbedding(self.time_emb_dim)
#        
#        self.time_emb_add = 0
        

        self.down1 = ResNetBlock(in_channels, init_features)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.down2 = ResNetBlock(init_features, init_features*2)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.down3 = ResNetBlock(init_features*2, init_features*4)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.down4 = ResNetBlock(init_features*4, init_features*8)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.bottleneck1 = ResNetBlock(init_features*8, init_features*16)
        #self.attn = SelfAttention(init_features * 16)

        self.up4 = ResNetBlock(init_features*16, init_features*8)
        self.upconv4 = nn.ConvTranspose2d(init_features*16, init_features*8, kernel_size=2, stride=2, padding=1, output_padding=1)
        self.up3 = ResNetBlock(init_features*8, init_features*4)
        self.upconv3 = nn.ConvTranspose2d(init_features*8, init_features*4, kernel_size=2, stride=2)
        self.up2 = ResNetBlock(init_features*4, init_features*2)
        self.upconv2 = nn.ConvTranspose2d(init_features*4, init_features*2, kernel_size=2, stride=2)
        self.up1 = ResNetBlock(init_features*2, init_features)
        self.upconv1 = nn.ConvTranspose2d(init_features*2, init_features, kernel_size=2, stride=2)

        self.final_conv = nn.Conv2d(in_channels=init_features, out_channels=out_channels, kernel_size=3, padding=1)

    def forward(self, x):
#        enc_time_emb = self.time_embedding(t)
#        dec_time_emb = self.time_embedding(t - self.time_emb_add)
        #time_emb_lookup = self.time_embedding(t) # Look up the vectors
        #time_emb = self.time_mlp(time_emb_lookup)

        #x_t = x_t + 0.1*torch.rand_like(x_t)
        x = x.float()
        
        enc1 = self.down1(x)
        enc2 = self.down2(self.pool1(enc1))
        enc3 = self.down3(self.pool2(enc2))
        enc4 = self.down4(self.pool3(enc3))

        bottleneck = self.bottleneck1(self.pool4(enc4))
        #bottleneck = self.attn(bottleneck)

        dec4 = self.upconv4(bottleneck)
        if dec4.shape[2:] != enc4.shape[2:]:
            dec4 = F.interpolate(dec4, size=enc4.shape[2:], mode='bilinear', align_corners=False)
        dec4 = torch.cat([dec4, enc4], dim=1)
        dec4 = self.up4(dec4)
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat([dec3, enc3], dim=1)
        dec3 = self.up3(dec3)
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat([dec2, enc2], dim=1)
        dec2 = self.up2(dec2)
        dec1 = self.upconv1(dec2)
        dec1 = F.interpolate(dec1, size=enc1.shape[2:], mode='bilinear', align_corners=False) 
        dec1 = torch.cat([dec1, enc1], dim=1)
        dec1 = self.up1(dec1)
        
        return self.final_conv(dec1)



class UNet_DM(nn.Module):
    def __init__(self, total_timesteps = 10, in_channels=2, out_channels=1, init_features=64):
        super(UNet_DM, self).__init__()

        self.time_emb_dim = 512
        self.time_embedding = TimeEmbedding(self.time_emb_dim)

#        self.time_embedding = nn.Embedding(total_timesteps, self.time_emb_dim) 
#        self.time_mlp = nn.Sequential

        self.down1 = ResNetBlock(in_channels, init_features, self.time_emb_dim)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.down2 = ResNetBlock(init_features, init_features*2, self.time_emb_dim)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.down3 = ResNetBlock(init_features*2, init_features*4, self.time_emb_dim)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.down4 = ResNetBlock(init_features*4, init_features*8, self.time_emb_dim)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.bottleneck1 = ResNetBlock(init_features*8, init_features*16, self.time_emb_dim)
        self.attn = SelfAttention(init_features * 16)

        self.up4 = ResNetBlock(init_features*16, init_features*8, self.time_emb_dim)
        self.upconv4 = nn.ConvTranspose2d(init_features*16, init_features*8, kernel_size=2, stride=2, padding=1, output_padding=1)
        self.up3 = ResNetBlock(init_features*8, init_features*4, self.time_emb_dim)
        self.upconv3 = nn.ConvTranspose2d(init_features*8, init_features*4, kernel_size=2, stride=2)
        self.up2 = ResNetBlock(init_features*4, init_features*2, self.time_emb_dim)
        self.upconv2 = nn.ConvTranspose2d(init_features*4, init_features*2, kernel_size=2, stride=2)
        self.up1 = ResNetBlock(init_features*2, init_features, self.time_emb_dim)
        self.upconv1 = nn.ConvTranspose2d(init_features*2, init_features, kernel_size=2, stride=2)

        self.final_conv = nn.Conv2d(in_channels=init_features, out_channels=out_channels, kernel_size=3, padding=1)

    def forward(self, x_t, cond, t):
        time_emb = self.time_embedding(t)
        
        #x_t = x_t + 0.1*torch.rand_like(x_t)
        x = torch.cat([x_t, cond], dim=1).float()
        
        enc1 = self.down1(x, time_emb)
        enc2 = self.down2(self.pool1(enc1), time_emb)
        enc3 = self.down3(self.pool2(enc2), time_emb)
        enc4 = self.down4(self.pool3(enc3), time_emb)

        bottleneck = self.bottleneck1(self.pool4(enc4), time_emb)
        bottleneck = self.attn(bottleneck)

        dec4 = self.upconv4(bottleneck)
        if dec4.shape[2:] != enc4.shape[2:]:
            dec4 = F.interpolate(dec4, size=enc4.shape[2:], mode='bilinear', align_corners=False)
        dec4 = torch.cat([dec4, enc4], dim=1)
        dec4 = self.up4(dec4, time_emb)
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat([dec3, enc3], dim=1)
        dec3 = self.up3(dec3, time_emb)
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat([dec2, enc2], dim=1)
        dec2 = self.up2(dec2, time_emb)
        dec1 = self.upconv1(dec2)
        dec1 = F.interpolate(dec1, size=enc1.shape[2:], mode='bilinear', align_corners=False) 
        dec1 = torch.cat([dec1, enc1], dim=1)
        dec1 = self.up1(dec1, time_emb)
        
        return self.final_conv(dec1)
#
