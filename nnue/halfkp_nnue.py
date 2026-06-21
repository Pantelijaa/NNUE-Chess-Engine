import chess
import torch
import torch.nn as nn

class SquaredClippedRelu(nn.Module):
    def forward(self, x):
        return torch.clamp(x, min=0.0, max=1.0) ** 2


class HalfKPNNUE(nn.Module):
    def __init__(self, input_features=41024, transformer_outputs=256):
        super(HalfKPNNUE, self).__init__()

        self.feature_transformer = nn.Linear(input_features, transformer_outputs)
        self.ft_activation = SquaredClippedRelu()

        self.hidden_layers = nn.Sequential(
            nn.Linear(transformer_outputs * 2, 32),
            SquaredClippedRelu(),
            nn.Linear(32, 32),
            SquaredClippedRelu(),
            nn.Linear(32, 1) # output
        )

    def forward(self, white_features, black_features, stm):
        white_t = self.ft_activation(self.feature_transformer(white_features))
        black_t = self.ft_activation(self.feature_transformer(black_features))

        # Iz white perspektive
        x_white = torch.cat((white_t, black_t), dim=1)
        out_white = self.hidden_layers(x_white)

        # Iz black perspektive
        x_black = torch.cat((black_t, white_t), dim=1)
        out_black = self.hidden_layers(x_black)

        stm_mask = stm.view(-1, 1).float()
        output = stm_mask * out_white + (1 - stm_mask) * out_black
        output = output - (stm_mask * out_black + (1 - stm_mask) * out_white)

        return output

    def forward_from_accumulator(self, white_acc, black_acc):
        w_activated = self.ft_activation(white_acc)
        b_activated = self.ft_activation(black_acc)
        x = torch.cat((w_activated, b_activated), 1)
        return self.hidden_layers(x)
