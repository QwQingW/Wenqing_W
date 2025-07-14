import torch
import torch.nn as nn
import torch.nn.functional as F

class RSDTransformer(nn.Module):
    def __init__(self, input_dim, d_model, nhead, num_layers, seq_len, pred_len):
        super().__init__()
        self.embedding = nn.Linear(input_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout = 0.1, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.trend_layer = nn.Sequential(
            nn.Linear(seq_len, 2 * pred_len),
            nn.ReLU(),
            nn.Linear(2 * pred_len, pred_len)
        )
        # self.seasonal_layer = nn.Linear(seq_len, pred_len)
        self.seasonal_layer = nn.Linear(d_model, pred_len)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def forward(self, x):

        trend = x.mean(dim=2)
        seasonal = x - x.mean(dim=2, keepdim=True)

        trend_out = self.trend_layer(trend)

        x_emb = self.embedding(seasonal)
        x_enc = self.encoder(x_emb)
        seasonal_out = self.seasonal_layer(x_enc[:, -1, :])


        return trend_out + seasonal_out

class DecomposedTransformer(nn.Module):
    def __init__(self, input_dim, d_model, nhead, num_layers, pred_len, kernel_size=25):
        super().__init__()
        self.pred_len = pred_len
        self.d_model = d_model
        self.kernel_size = kernel_size
        self.embedding = nn.Linear(input_dim, d_model)

        # 时间序列分解：滑动平均
        self.moving_avg = nn.AvgPool1d(kernel_size=kernel_size, stride=1, padding=kernel_size//2)

        # Trend 预测分支（单层MLP）
        self.trend_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, pred_len)
        )

        # Seasonal 编码器（Transformer）
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Seasonal 输出头
        self.seasonal_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, pred_len)
        )

        # self.alpha = nn.Parameter(torch.tensor(0.5))

        self.alpha_layer = nn.Sequential(
            nn.Linear(d_model, 1),  # 输入当前时刻的特征，输出趋势占比
            nn.Sigmoid()
        )

    def decompose(self, x):
        B, T, D = x.shape
        trend = F.avg_pool1d(
            x.transpose(1, 2), kernel_size=self.kernel_size, stride=1, padding=self.kernel_size // 2
        ).transpose(1, 2)

        if trend.shape[1] > T:
            trend = trend[:, :T, :]
        elif trend.shape[1] < T:
            trend = F.pad(trend, (0, 0, 0, T - trend.shape[1]))

        seasonal = x - trend
        return trend, seasonal

    def forward(self, x):

        x_embed = self.embedding(x)
        trend, seasonal = self.decompose(x_embed)

        # Trend 分支
        trend_input = trend[:, -5, :]
        trend_out = self.trend_head(trend_input)

        # Seasonal 分支
        seasonal_encoded = self.encoder(seasonal)
        seasonal_out = self.seasonal_head(seasonal_encoded[:, -1, :])

        alpha = self.alpha_layer(seasonal_encoded[:, -5, :])  # [B, 1]
        alpha = alpha.squeeze(-1)  # [B]

        # 预测值融合
        out = alpha.unsqueeze(1) * trend_out + (1 - alpha).unsqueeze(1) * seasonal_out  # [B, pred_len]

        return out, trend_out, seasonal_out, alpha  #