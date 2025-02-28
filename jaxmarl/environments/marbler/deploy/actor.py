import torch.nn as nn

class RNNActor(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim

        self.Dense_0 = nn.Linear(input_dim, hidden_dim)
        self.GRUCell_0 = nn.GRUCell(hidden_dim, hidden_dim)
        self.Dense_1 = nn.Linear(hidden_dim, output_dim)

    def forward(self, input, hidden): 
        embedding = self.Dense_0(input)
        hidden = self.GRUCell_0(embedding, hidden)
        output = self.Dense_1(hidden)

        return output, hidden
