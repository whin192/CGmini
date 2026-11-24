import torch
import torch.nn as nn
import torch.nn.functional as F


class Attribute(nn.Module):
    def __init__(self, input_dims, hidden_dims, emb_dims, alpha, beta, dropout, device):
        super(Attribute, self).__init__()
        self.mlp = nn.Sequential(nn.Linear(input_dims, hidden_dims, bias=False),
                                 nn.ReLU(inplace=True),
                                 nn.Dropout(dropout),
                                 nn.Linear(hidden_dims, emb_dims, bias=False))
        self.classify = nn.Sequential(nn.Linear(emb_dims, emb_dims, bias=False))

        self.alpha = alpha
        self.beta = beta
        self.dropout = dropout
        self.device = device

    def express(self, h, x_dis):
        coe2 = 1.0 / self.beta
        res = torch.mm(torch.transpose(h, 0, 1), h)
        inv = torch.inverse(torch.eye(h.shape[1]).to(self.device) + coe2 * res)
        res = torch.mm(inv, res)
        res = coe2 * h - coe2 * coe2 * torch.mm(h, res)
        tmp = torch.mm(torch.transpose(h, 0, 1), res)
        part1 = torch.mm(h, tmp)
        part2 = (- self.alpha / 2) * torch.mm(x_dis, res)

        emb_hom = part1 + part2
        return emb_hom

    # def gen_s(self, h, x_dis):
    #     res = torch.mm(h, torch.transpose(h, 0, 1))
    #     part1 = res - self.alpha / 2 * x_dis
    #     part2 = torch.inverse(res + self.beta * torch.eye(h.shape[0]).to(self.device))
    #     s = torch.mm(part1, part2)
    #     return s

    def forward(self, fea, fea_dis):
        h = self.mlp(fea)
        z = self.express(h, fea_dis)
        z_s = self.classify(z)
        return z_s


class ScaledDotProductAttention(nn.Module):
    def __init__(self, temperature, attn_dropout=0.1):
        super(ScaledDotProductAttention, self).__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, mask=None):
        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))

        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)

        attn = self.dropout(F.softmax(attn, dim=-1))
        output = torch.matmul(attn, v)

        return output, attn


class MultiHeadAttention(nn.Module):
    def __init__(self, n_head, channels, dropout=0.1):
        super(MultiHeadAttention, self).__init__()

        self.n_head = n_head
        self.channels = channels
        d_q = d_k = d_v = channels // n_head

        self.w_qs = nn.Linear(channels, channels, bias=False)
        self.w_ks = nn.Linear(channels, channels, bias=False)
        self.w_vs = nn.Linear(channels, channels, bias=False)
        self.fc = nn.Linear(channels, channels, bias=False)

        self.attention = ScaledDotProductAttention(temperature=d_k ** 0.5)

        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        n_head = self.n_head
        d_q = d_k = d_v = self.channels // n_head
        B_q = q.size(0)
        N_q = q.size(1)

        B_k = k.size(0)
        N_k = k.size(1)

        B_v = v.size(0)
        N_v = v.size(1)

        residual = q
        q = self.w_qs(q).view(B_q, N_q, n_head, d_q)
        k = self.w_ks(k).view(B_k, N_k, n_head, d_k)
        v = self.w_vs(v).view(B_v, N_v, n_head, d_v)

        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        if mask is not None:
            mask = mask.unsqueeze(1)

        q, attn = self.attention(q, k, v, mask=mask)

        q = q.transpose(1, 2).contiguous().view(B_q, N_q, -1)
        q = self.fc(q)
        q = q + residual

        return q, attn


class TransLayer(nn.Module):
    def __init__(self, n_head, channels, use_patch_attn=True, dropout=0.1):
        super(TransLayer, self).__init__()
        # intra
        self.node_norm = nn.LayerNorm(channels)
        self.node_transformer = MultiHeadAttention(n_head, channels, dropout)
        self.node_ffn = FFN(channels, dropout)

        # inter
        self.patch_norm = nn.LayerNorm(channels)
        self.patch_transformer = MultiHeadAttention(n_head, channels, dropout)
        self.patch_ffn = FFN(channels, dropout)

        self.fuse_lin = nn.Linear(2 * channels, channels)
        self.use_patch_attn = use_patch_attn
        self.attn = None

    def forward(self, x, attn_mask=None):
        x = self.node_norm(x)
        hp, attn = self.node_transformer(x, x, x, attn_mask)
        hp = self.node_ffn(hp)

        if self.use_patch_attn:
            p = self.patch_norm(hp.mean(dim=1, keepdim=False)).unsqueeze(0)
            p, _ = self.patch_transformer(p, p, p)
            p = self.patch_ffn(p).permute(1, 0, 2)

            p = p.repeat(1, hp.shape[1], 1)
            z = torch.cat([hp, p], dim=2)

        final_z = z.mean(dim=0, keepdim=False)
        return final_z


class FFN(nn.Module):
    def __init__(self, channels, dropout=0.1):
        super(FFN, self).__init__()
        self.lin1 = nn.Linear(channels, channels)  # position-wise
        self.lin2 = nn.Linear(channels, channels)  # position-wise
        self.layer_norm = nn.LayerNorm(channels, eps=1e-6)
        self.Dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        x = self.layer_norm(x)
        x = self.Dropout(x)
        x = F.relu(self.lin1(x))
        x = self.lin2(x) + residual

        return x


class Topology(torch.nn.Module):
    def __init__(self, input_dims, hidden_dims, num_view, num_classes, dropout):
        super(Topology, self).__init__()
        self.n_head = 4
        self.layers = 1
        self.num_view = num_view
        self.mlp = nn.Sequential(nn.Linear(input_dims, hidden_dims, bias=False),
                                 nn.ReLU(inplace=True),
                                 nn.Dropout(dropout),
                                 nn.Linear(hidden_dims, hidden_dims, bias=False))

        self.Cross_path_transformer = nn.ModuleList()
        for v in range(self.layers):
            self.Cross_path_transformer.append(TransLayer(self.n_head, hidden_dims, use_patch_attn=True, dropout=dropout))

        self.classifier = nn.Linear(hidden_dims * 2, num_classes, bias=False)

    def forward(self, x, adj_list):
        x = self.mlp(x)
        agg_x = []
        for v in range(self.num_view):
            agg_x.append(adj_list[v] @ x)

        stack_x = torch.stack(agg_x)
        for v in range(self.layers):
            z = self.Cross_path_transformer[v](stack_x)
        pred = self.classifier(z)
        return pred


class CGmini(nn.Module):
    def __init__(self, input_dims, hidden_dims, num_view, num_classes, dropout, alpha, beta, device):
        super(CGmini, self).__init__()
        self.alpha = alpha
        self.beta = beta

        self.input_dims = input_dims
        self.hidden_dims = hidden_dims
        self.num_views = num_view

        self.device = device
        self.relu = nn.ReLU(inplace=True)
        self.dropout = dropout

        self.classifier = nn.ModuleList()

        self.attr_level = Attribute(input_dims, hidden_dims, num_classes, self.alpha, self.beta, self.dropout, self.device)
        self.topo_level = Topology(input_dims, hidden_dims, num_view, num_classes, dropout)

    def forward(self, features, features_dis, adj_hat_list):
        attr_z = self.attr_level(features, features_dis)

        topo_z = self.topo_level(features, adj_hat_list)
        return attr_z, topo_z