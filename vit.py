"""
This code was based on the file vit.py (https://github.com/lucidrains/vit-pytorch/blob/main/vit_pytorch/vit.py)
from the lucidrains/vit-pytorch library (https://github.com/lucidrains/vit-pytorch).

The original license is included below:

BSD 3-Clause License

Copyright (c) Soumith Chintala 2016,
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

* Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""
import torch
from torch import nn
from avalanche.models import BaseModel, MultiHeadClassifier, IncrementalClassifier, MultiTaskModule, DynamicModule

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

# helpers

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

# classes

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class T_block(nn.Module):
    def __init__(self, dim, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.attn = PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout))
        self.ff = PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
    def forward(self, x):
        x = self.attn(x) + x
        x = self.ff(x) + x
        return x


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(T_block(dim, heads, dim_head, mlp_dim, dropout))

    def forward(self, x):
        for blk in self.layers:
            x = blk(x)
        return x


class Encoder(nn.Module):
    def __init__(self, patch_height, patch_width, patch_dim, dim, num_patches, emb_dropout):
        super().__init__()
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_height, p2=patch_width),
            # nn.BatchNorm1d(num_patches),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            # nn.BatchNorm1d(num_patches),
            nn.LayerNorm(dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

    def forward(self, img):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b=b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        return x


class ViT(nn.Module):
    """
    MODIFY: remove Linear in mlp_head
    """
    def __init__(self, *, image_size, patch_size,
                 # num_classes,
                 dim, depth, heads, mlp_dim,
                 pool = 'cls', channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0.):
        super().__init__()
        self.dim = dim
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            # nn.Linear(dim, num_classes)
        )

    def forward(self, img):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        x = self.transformer(x)

        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)
        return self.mlp_head(x)

    @property
    def output_size(self):
        return self.dim


class MultiHeadRPS_net_ViT(nn.Module):

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.image_size = 128
        self.patch_size = 16
        self.dim = 384
        # self.depth = 9
        self.heads = 16
        self.mlp_dim = 1536
        self.block_dropout = 0.1
        self.emb_dropout = 0.1
        self.pool = 'cls'
        self.channels = 3
        self.dim_head = 64

        # self.to_latent = nn.Identity()
        # self.mlp_head = nn.Sequential(
        #     nn.LayerNorm(self.dim),
        # )

        self.final_layers = []
        self.init(None)

    def init(self, best_path):

        image_height, image_width = pair(self.image_size)
        patch_height, patch_width = pair(self.patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = self.channels * patch_height * patch_width
        assert self.pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.encoder = nn.ModuleList()    # encoder
        self.l1 = nn.ModuleList()
        self.l2 = nn.ModuleList()
        self.l3 = nn.ModuleList()
        self.l4 = nn.ModuleList()
        self.l5 = nn.ModuleList()
        self.l6 = nn.ModuleList()
        self.l7 = nn.ModuleList()
        self.l8 = nn.ModuleList()
        self.l9 = nn.ModuleList()

        for i in range(self.args.M):
            self.encoder.append(Encoder(patch_height, patch_width, patch_dim, self.dim, num_patches, self.emb_dropout))
            self.l1.append(T_block(self.dim, self.heads, self.dim_head, self.mlp_dim, self.block_dropout))
            self.l2.append(T_block(self.dim, self.heads, self.dim_head, self.mlp_dim, self.block_dropout))
            self.l3.append(T_block(self.dim, self.heads, self.dim_head, self.mlp_dim, self.block_dropout))
            self.l4.append(T_block(self.dim, self.heads, self.dim_head, self.mlp_dim, self.block_dropout))
            self.l5.append(T_block(self.dim, self.heads, self.dim_head, self.mlp_dim, self.block_dropout))
            self.l6.append(T_block(self.dim, self.heads, self.dim_head, self.mlp_dim, self.block_dropout))
            self.l7.append(T_block(self.dim, self.heads, self.dim_head, self.mlp_dim, self.block_dropout))
            self.l8.append(T_block(self.dim, self.heads, self.dim_head, self.mlp_dim, self.block_dropout))
            self.l9.append(T_block(self.dim, self.heads, self.dim_head, self.mlp_dim, self.block_dropout))

        # add mx
        self.l4.append(T_block(self.dim, self.heads, self.dim_head, self.mlp_dim, self.block_dropout))
        self.l6.append(T_block(self.dim, self.heads, self.dim_head, self.mlp_dim, self.block_dropout))
        self.l8.append(T_block(self.dim, self.heads, self.dim_head, self.mlp_dim, self.block_dropout))

        # final layer
        if len(self.final_layers) < 1:
            if self.args.return_task_id:    # task-IL
                for i in range(self.args.num_train_task):
                    exec(f"self.final_layer{i+1} = nn.Linear({self.dim}, self.args.class_per_task)")    # 10
                    exec(f"self.final_layers.append(self.final_layer{i+1})")

                exec(f"self.final_layer_test = nn.Linear({self.dim}, self.args.num_test_class)")  # 10
                exec(f"self.final_layers.append(self.final_layer_test)")

            else:    # class-IL
                exec(f"self.final_layer1 = nn.Linear({self.dim}, self.args.class_per_task*self.args.num_train_task)")    # 100
                exec(f"self.final_layers.append(self.final_layer1)")

                exec(f"self.final_layer_test = nn.Linear({self.dim}, self.args.num_test_class)")  # 10
                exec(f"self.final_layers.append(self.final_layer_test)")

        self.cuda()

    def freeze_feature_extractor(self):
        params_set = [self.encoder, self.l1, self.l2, self.l3, self.l4, self.l5, self.l6, self.l7, self.l8, self.l9]
        for j, params in enumerate(params_set):
            for i, param in enumerate(params):
                param.requires_grad = False

    def forward(self, img, path, last):
        # x = self.to_patch_embedding(img)
        # b, n, _ = x.shape
        #
        # cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b=b)
        # x = torch.cat((cls_tokens, x), dim=1)
        # x += self.pos_embedding[:, :(n + 1)]
        # x = self.dropout(x)

        # x = self.transformer(x)

        x = img

        y = self.encoder[0](x)
        for j in range(1, self.args.M):
            if (path[0][j] == 1):
                y += self.encoder[j](x)
        x = y

        y = self.l1[0](x)
        for j in range(1, self.args.M):
            if (path[1][j] == 1):
                y += self.l1[j](x)
        x = y

        y = self.l2[0](x)
        for j in range(1, self.args.M):
            if (path[2][j] == 1):
                y += self.l2[j](x)
        x = y

        y = self.l3[0](x)
        for j in range(1, self.args.M):
            if (path[3][j] == 1):
                y += self.l3[j](x)
        x = y

        y = self.l4[-1](x)
        for j in range(self.args.M):
            if (path[4][j] == 1):
                y += self.l4[j](x)
        x = y

        y = self.l5[0](x)
        for j in range(1, self.args.M):
            if (path[5][j] == 1):
                y += self.l5[j](x)
        x = y

        y = self.l6[-1](x)
        for j in range(self.args.M):
            if (path[6][j] == 1):
                y += self.l6[j](x)
        x = y

        y = self.l7[0](x)
        for j in range(1, self.args.M):
            if (path[7][j] == 1):
                y += self.l7[j](x)
        x = y

        y = self.l8[-1](x)
        for j in range(self.args.M):
            if (path[8][j] == 1):
                y += self.l8[j](x)
        x = y

        y = self.l9[0](x)
        for j in range(1, self.args.M):
            if (path[9][j] == 1):
                y += self.l9[j](x)
        x = y

        x = x.mean(dim=1) if self.pool == 'mean' else x[:, 0]

        # x = self.to_latent(x)
        # x = self.mlp_head(x)

        if type(last) is int:
            x = self.final_layers[last](x)
        else:
            o = []
            for x_idx in range(x.shape[0]):
                if last[x_idx] >= len(self.final_layers):  # for few-shot test
                    idx = -1
                else:  # for continual train
                    idx = last[x_idx]
                o.append(self.final_layers[idx](x[x_idx: x_idx + 1]))  # forward classifier sample by sample
            x = torch.cat(o)

        return x

    @property
    def output_size(self):
        return self.dim


if __name__ == '__main__':

    def get_parameter_number(model):
        total_num = sum(p.numel() for p in model.parameters())
        trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
        return {'Total': total_num, 'Trainable': trainable_num}

    # from vit_pytorch import ViT
    v = ViT(
        image_size=128,     # org: 256
        patch_size=16,      # org: 32
        # num_classes=2,
        dim=512,   # org: 1024  512: same as resnet-18
        depth=5,        # org: 6
        heads=8,        # org: 16
        mlp_dim=1024,       # org: 2048
        dropout=0.1,
        emb_dropout=0.1
    )
    # v = ViT(      # same param size (10MB) as resnet-18
    #     image_size=128,     # org: 256
    #     patch_size=16,      # org: 32
    #     # num_classes=2,
    #     dim=512,   # org: 1024  512: same as resnet-18
    #     depth=5,        # org: 6
    #     heads=8,        # org: 16
    #     mlp_dim=1024,       # org: 2048
    #     dropout=0.1,
    #     emb_dropout=0.1
    # )
    # img = torch.randn(1, 3, 256, 256)
    # preds = v(img)  # (1, 1000)

    d = get_parameter_number(v)
    print(d)

    print(f'Total number of parameters: {d["Total"] / 1024 / 1024:.2f}MB, '
          f'memory size: {d["Total"] * 4 / 1024 / 1024:.2f}MB')
    print(f'Total number of trainable parameters: {d["Trainable"] / 1024 / 1024:.2f}MB, '
          f'memory size: {d["Trainable"] * 4 / 1024 / 1024:.2f}MB')


    # 10.43MB
