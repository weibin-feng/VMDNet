import torch
import torch.nn as nn
from utils.vmd_utils import vmd_batch
import math


class VMDModule(nn.Module):
    def __init__(self, K=4, alpha=2000, tau=0.0, tol=1e-7, device='cuda'):
        super(VMDModule, self).__init__()
        self.vmd_params = {
            "alpha": alpha,
            "tau": tau,
            "K": K,
            "DC": False,
            "init": 1,
            "tol": tol,
        }
        self.device = device


    def forward(self, x):
        """
        x: [B, T, C] 
        return:
            u: [B, K, T, C]
            omega: [B, K]
        """
        x = x.permute(0, 2, 1)  # [B, C, T]

        u, _, omega = vmd_batch(x, **self.vmd_params, device=x.device)
        return u, omega



class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        pe = torch.zeros(max_len, d_model).float()
        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        seq_len = x.size(1)
        return self.pe[:, :seq_len].to(x.device)


class TokenEmbedding(nn.Module):
    def __init__(self, enc_in, d_model):
        super(TokenEmbedding, self).__init__()
        self.tokenConv = nn.Conv1d(in_channels=enc_in, out_channels=d_model,
                                   kernel_size=3, padding=1, padding_mode='circular', bias=False)
        nn.init.kaiming_normal_(self.tokenConv.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.tokenConv(x)
        x = x.transpose(1, 2)
        return x


class TimeFeatureEmbedding(nn.Module):
    def __init__(self, d_model, embed_type='timeF', freq='h'):
        super(TimeFeatureEmbedding, self).__init__()
        freq_map = {'h': 4, 't': 5, 's': 6, 'm': 1, 'a': 1, 'w': 2, 'd': 3, 'b': 3}
        d_inp = freq_map.get(freq, 4)
        self.embed = nn.Linear(d_inp, d_model, bias=False)

    def forward(self, x):
        return self.embed(x)


class MultiModeDataEmbedding(nn.Module):
    def __init__(self, enc_in, d_model, embed_type='timeF', freq='h', dropout=0.1, n_modes=4,
                 use_freq_embed: bool = True):
        super(MultiModeDataEmbedding, self).__init__()
        self.n_modes = n_modes
        self.d_model = d_model
        self.use_freq_embed = use_freq_embed

        self.token_embeddings = nn.ModuleList([
            TokenEmbedding(enc_in, d_model) for _ in range(n_modes)
        ])
        if self.use_freq_embed:
            self.freq_embeddings = nn.ModuleList([
                nn.Linear(1, d_model, bias=False) for _ in range(n_modes)
            ])
        else:
            self.freq_embeddings = None
        self.position_embedding = PositionalEmbedding(d_model=d_model, max_len=5000)

        if embed_type == 'timeF':
            self.temporal_embedding = TimeFeatureEmbedding(d_model, freq=freq)
        else:
            raise NotImplementedError("Only embed_type='timeF' is supported.")

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark, x_omega):
        B, K, seq_len, C = x.shape
        out_list = []
        for i in range(K):
            mode_x = x[:, i, :, :]
            val_emb = self.token_embeddings[i](mode_x)                # [B,T,d]
            pos_emb = self.position_embedding(val_emb)                # [B,T,d]
            time_emb_x = self.temporal_embedding(x_mark.to(x.device)) # [B,T,d]

            if self.use_freq_embed:
                freq_i = x_omega[:, i].unsqueeze(-1)                  # [B,1]
                freq_emb = self.freq_embeddings[i](freq_i.to(x.device))      # [B,d]
                freq_emb_x = freq_emb.unsqueeze(1).repeat(1, seq_len, 1)     # [B,T,d]
                mode_out = val_emb + pos_emb + time_emb_x + freq_emb_x
            else:
                mode_out = val_emb + pos_emb + time_emb_x

            out_list.append(mode_out)

        out = torch.stack(out_list, dim=1)  # [B,K,T,d]
        return self.dropout(out)



class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, dilation, padding, dropout,activation='gelu'):
        super(TemporalBlock, self).__init__()

        if activation == 'relu':
            self.activation_fn = nn.ReLU()
        elif activation == 'gelu':
            self.activation_fn = nn.GELU()
        else:
            raise ValueError(f"Unsupported activation function: {activation}. Choose 'relu' or 'gelu'.")

        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size,
                               stride=stride, padding=padding, dilation=dilation)
        self.chomp1 = Chomp1d(padding)
        self.dropout1 = nn.Dropout(dropout)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size,
                               stride=stride, padding=padding, dilation=dilation)
        self.chomp2 = Chomp1d(padding)
        self.dropout2 = nn.Dropout(dropout)

        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None
        self.norm = nn.LayerNorm(out_channels)

    def forward(self, x):
        out = self.conv1(x)
        out = self.chomp1(out)
        out = self.activation_fn(out) 
        out = self.dropout1(out)

        out = self.conv2(out)
        out = self.chomp2(out)
        out = self.activation_fn(out)
        out = self.dropout2(out)

        res = x if self.downsample is None else self.downsample(x)
        out = out + res
        out = self.norm(out.transpose(1, 2)).transpose(1, 2)
        return out


class TCNDecoder(nn.Module):
    def __init__(self, input_size, output_size, num_channels, kernel_size=3, dropout=0.2, pred_len=96):
        super(TCNDecoder, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = input_size if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1,
                                     dilation=dilation_size, padding=(kernel_size - 1) * dilation_size,
                                     dropout=dropout)]

        self.network = nn.Sequential(*layers)
        self.proj = nn.Linear(num_channels[-1], output_size)
        self.pred_len = pred_len

    def forward(self, x):
        x = x.transpose(1, 2)
        out = self.network(x)
        out = out.transpose(1, 2)
        out = out[:, -self.pred_len:, :]
        out = self.proj(out)
        return out

class TransformerDecoder(nn.Module):
    def __init__(
        self,
        input_size,
        d_model,
        nhead,
        num_layers,
        dim_feedforward,
        dropout,
        pred_len
    ):
        super().__init__()

        self.input_proj = nn.Linear(input_size, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.output_proj = nn.Linear(d_model, 1)
        self.pred_len = pred_len

    def forward(self, x):
        """
        x: [B, T, input_size]
        return: [B, pred_len, 1]
        """
        x = self.input_proj(x)       # [B,T,d_model]
        x = self.encoder(x)          # [B,T,d_model]
        x = x[:, -self.pred_len:, :] # take last pred_len
        out = self.output_proj(x)    # [B,pred_len,1]
        return out


class MultiModeTCNDecoder(nn.Module):
    def __init__(self, d_model, pred_len, n_modes=4, hidden_dims=[64, 64], dropout=0.2):
        super(MultiModeTCNDecoder, self).__init__()

        self.decoders = nn.ModuleList([
            TCNDecoder(input_size=d_model, output_size=1, num_channels=hidden_dims, dropout=dropout, pred_len=pred_len)
            for _ in range(n_modes)
        ])

    def forward(self, x):
        outs = []
        for i, dec in enumerate(self.decoders):
            mode_x = x[:, i, :, :]
            out = dec(mode_x)
            outs.append(out)

        outs = torch.stack(outs, dim=1)
        return outs


class FusionModule(nn.Module):
    def __init__(self, n_modes, pred_len, output_size=1, method='sum', ffn_hidden_dim=64):
        """
        n_modes: number of modes K
        pred_len: prediction length
        output_size: final output dimension (usually 1 for single-variable prediction)
        method: fusion method:
            - 'sum': sum all modes
            - 'linear': learnable scalar per mode
            - 'ffn': use FFN for fusion
        ffn_hidden_dim: hidden dimension for FFN (only used when method='ffn')
        """
        super(FusionModule, self).__init__()
        self.method = method.lower()
        self.n_modes = n_modes
        self.pred_len = pred_len
        self.output_size = output_size

       

        if self.method == 'linear':
            # learnable scalar per mode
            self.weights = nn.Parameter(torch.ones(n_modes))  # shape [K]
        elif self.method == 'ffn':
            # Input to FFN will be [B, K * pred_len * output_size]
            # Output from FFN will be [B, pred_len * output_size]
            input_ffn_dim = n_modes * pred_len * output_size
            self.ffn = nn.Sequential(
                nn.Linear(input_ffn_dim, ffn_hidden_dim),
                nn.GELU(), 
                nn.Linear(ffn_hidden_dim, pred_len * output_size)
            )

    def forward(self, x):
        """
        x: [B, K, pred_len, 1]
        return: [B, pred_len, 1]
        """
        if self.method == 'sum':
            return x.sum(dim=1)

        elif self.method == 'linear':
            # weights: [K] -> [1, K, 1, 1] for broadcasting
            w = self.weights.view(1, self.n_modes, 1, 1)
            x_weighted = x * w  # [B, K, pred_len, 1]
            return x_weighted.sum(dim=1)

        elif self.method == 'ffn':
            # Flatten the input for FFN: [B, K, pred_len, 1] -> [B, K * pred_len * 1]
            # Reshape x to [B, K * pred_len * output_size]
            x_flat = x.view(x.size(0), -1)

            # Pass through FFN
            ffn_out = self.ffn(x_flat) # [B, pred_len * output_size]

            # Reshape back to [B, pred_len, output_size]
            return ffn_out.view(x.size(0), self.pred_len, self.output_size)

        else:
            raise ValueError(f"Unsupported fusion method: {self.method}")



class VMDNet(nn.Module):
    def __init__(self, args):
        super(VMDNet, self).__init__()

        self.vmd = VMDModule(
            K=args.vmd_K, alpha=args.vmd_alpha, tau=args.vmd_tau, tol=args.vmd_tol, device=args.device
        )
        self.embedding = MultiModeDataEmbedding(
            enc_in=args.enc_in, d_model=args.d_model, embed_type=args.embed, freq=args.freq,
            dropout=args.dropout, n_modes=args.vmd_K,
            use_freq_embed=True
        )

        self.decoder = MultiModeTCNDecoder(
            d_model=args.d_model, pred_len=args.pred_len, n_modes=args.vmd_K,
            hidden_dims=args.tcn_hidden_dims, dropout=args.tcn_dropout
        )
        self.fusion = FusionModule(
            n_modes=args.vmd_K, pred_len=args.pred_len, output_size=1,
            method=args.fusion_method, ffn_hidden_dim=getattr(args, 'ffn_hidden_dim', 64)
        )

    def forward(self, x, x_mark, dec_inp=None, y_mark=None):
        u, omega = self.vmd(x)                      # u: [B,K,T,C], omega:[B,K]
        emb = self.embedding(u, x_mark, omega)      # emb: [B,K,T,d]
        dec_out = self.decoder(emb)             # [B,K,pred_len,1]
        output = self.fusion(dec_out)           # [B,pred_len,1]
        return output
