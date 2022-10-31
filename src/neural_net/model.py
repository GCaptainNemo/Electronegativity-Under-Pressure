import torch.nn as nn
import torch
from einops import rearrange
import torch.nn.functional as F
from src.settings.settings import TaskSettingObj


class SwishActivation(nn.Module):
    def __init__(self):
        # tanh is better than swish
        super(SwishActivation, self).__init__()

    def forward(self, x):
        return torch.sigmoid(x) * x


class ABTD(nn.Module):
    def __init__(self, i_dim, o_dim, use_attention=TaskSettingObj.baseline_is_use_attention):
        super(ABTD, self).__init__()
        # LBRD
        if use_attention:
            print("ABTD use attention")
            # https://github.com/somepago/saint/blob/main/models/model.py
            self.seq_block = nn.Sequential(
                Attention(input_dim=i_dim, output_dim=o_dim),
                nn.BatchNorm1d(o_dim),
                # layernorm + GeLU效果差于BN + Tanh
                # nn.ReLU(inplace=True), # the function is C0
                # nn.ELU(),         # the function is C1
                nn.Tanh(),  # the function is C2
                nn.Dropout() # default p=0.5
            )
        # origin
        else:
            print("LBTD not use attention")
            self.seq_block = nn.Sequential(
                nn.Linear(i_dim, o_dim),
                nn.BatchNorm1d(o_dim),
                # nn.ReLU(inplace=True), # the function is C0
                # nn.ELU(),         # the function is C1
                nn.Tanh(),          # the function is C2
                nn.Dropout()
            )

    def forward(self, x):
        return self.seq_block(x)


class DenseTransformerBlock(nn.Module):
    def __init__(self, input_dim=20, first_dim=30, second_dim=20):
        super(DenseTransformerBlock, self).__init__()
        self.seq_1 = nn.Sequential(
            nn.Linear(input_dim, first_dim),
            # nn.ReLU()
            # nn.ELU()
            nn.Tanh()
        )
        self.deep_block_1 = ABTD(first_dim, second_dim)
        self.deep_block_2 = ABTD(second_dim, 10)
        self.deep_block_3 = nn.Sequential(
            nn.BatchNorm1d(10),
            nn.Dropout(),
            nn.Linear(10, 1),
            # Attention(input_dim=10, output_dim=1),
        )
        self.wide_block_1 = nn.Linear(first_dim, 1)
        # self.wide_block_1 = Attention(input_dim=first_dim, output_dim=1)

    def forward(self, x):
        o = self.seq_1(x)
        o1 = self.wide_block_1(o)
        o2 = self.deep_block_3(self.deep_block_2(self.deep_block_1(o)))
        return o1 + o2


class DenseTransformer(nn.Module):
    def __init__(self, input_dim=20, first_dim=30, second_dim=20, layer_num=2):
        super(DenseTransformer, self).__init__()
        self.layer_num = layer_num
        self.input_dim = input_dim
        self.first_dim = first_dim
        self.second_dim = second_dim
        self.layers = nn.ModuleList(
            [DenseTransformerBlock(input_dim=self.input_dim + i,
                                   first_dim=self.first_dim,
                                   second_dim=self.second_dim)
             for i in range(self.layer_num)])

    def forward(self, x):
        input_ = x
        for i in range(len(self.layers)):
            out = self.layers[i](input_)
            input_ = torch.cat((input_, out), dim=1)
        return out


class MultiBranchModel(nn.Module):
    def __init__(self, input_dim_lst, first_dim=30, second_dim=20, layer_num=2):
        """
            Multi Branch Model
        """
        super(MultiBranchModel, self).__init__()
        self.partition_lst = [0]
        self.model_lst = nn.ModuleList([DenseTransformer(input_dim, first_dim, second_dim,
                                                         layer_num=layer_num) for input_dim in input_dim_lst])
        for val in input_dim_lst:
            self.partition_lst.append(self.partition_lst[-1] + val)

    def forward(self, x):
        res = torch.cat([model(x[:, self.partition_lst[i]:self.partition_lst[i + 1]])
                         for i, model in enumerate(self.model_lst)], dim=1)
        # print(res)
        return torch.mean(res, dim=1, keepdim=True)


class MultiBranchModelTest(nn.Module):
    def __init__(self, input_dim_lst, first_dim=30, second_dim=20, predict_flag=True):
        """
            Multi Branch Model
        """
        super(MultiBranchModelTest, self).__init__()
        self.partition_lst = [0]
        self.model_lst = nn.ModuleList([DenseTransformer(input_dim, first_dim, second_dim) for input_dim in input_dim_lst])
        for val in input_dim_lst:
            self.partition_lst.append(self.partition_lst[-1] + val)

    def forward(self, x):
        res = torch.cat([model(x[:, self.partition_lst[i]:self.partition_lst[i + 1]])
                         for i, model in enumerate(self.model_lst)], dim=1)
        return res


class Attention(nn.Module):
    def __init__(
        self,
        input_dim,  # input/output dimension
        output_dim,
        heads=8,
        dim_head=16,
    ):
        super().__init__()
        inner_dim = dim_head * heads  # 8 * 16 = 128
        self.heads = heads
        self.scale = dim_head ** -0.5  # 1/ 4
        self.to_qkv = nn.Linear(input_dim, inner_dim * 3, bias=False)
        self.to_out = nn.Linear(inner_dim, output_dim)

    def forward(self, x):
        h = self.heads
        q, k, v = self.to_qkv(x).chunk(3, dim=-1) # N x i -> N x 3heads*dim_head -> N x heads*dim_head
        # N x head x dim_head
        q, k, v = map(lambda t: rearrange(t, 'b (h d) -> b h d', h=h), (q, k, v))
        sim = torch.einsum('b h i, b h j -> b h i j', q, k) * self.scale
        attn = sim.softmax(dim=-1)  # key softmax
        out = torch.einsum('b h i j, b h j -> b h i', attn, v)
        out = rearrange(out, 'b h i -> b (h i)', h=h) # N x heads*dim_head
        out = self.to_out(out)
        return out
