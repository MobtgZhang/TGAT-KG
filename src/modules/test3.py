import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.nn import RGCNConv

class CapsuleLayer(nn.Module):
    def __init__(self, input_dim, output_dim, num_capsules, capsule_size):
        super(CapsuleLayer, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_capsules = num_capsules
        self.capsule_size = capsule_size

        # Define the linear transformation layer for each relation type
        self.relations = nn.ModuleList([
            nn.Linear(input_dim, output_dim * capsule_size) for _ in range(num_capsules)
        ])

        # Define the weight matrix for mapping the output capsule to the input capsule
        self.route_weights = nn.Parameter(torch.randn(num_capsules, input_dim, output_dim, capsule_size))

    def forward(self, x, edge_index, edge_type):
        batch_size = x.size(0)

        # Apply the linear transformation layer to each relation type
        capsules = []
        for relation in range(self.num_capsules):
            relation_weight = self.relations[relation](x)
            relation_weight = relation_weight.view(batch_size, self.output_dim, -1)
            capsules.append(relation_weight)

        # Reshape the capsules tensor to [batch_size, num_capsules=num_features, dim_capsule=-1]
        capsules_tensor = torch.stack(capsules, dim=-2)
        capsules_tensor = capsules_tensor.view(batch_size, self.num_capsules, -1)

        # Compute the weighted-sum between input capsules and output capsules
        # using the weight matrix and dynamic routing algorithm
        u_hat = torch.matmul(capsules_tensor[:, None, :, :], self.route_weights)
        u_hat = u_hat.squeeze()

        # Routing algorithm
        b = torch.zeros(batch_size, self.num_capsules, x.size(-1))

        num_iterations = 3
        for iteration in range(num_iterations):
            c = F.softmax(b, dim=1)
            s = (c[:, :, :, None] * u_hat[:, None, :, :]).sum(dim=-2)
            v = self.squash(s)

            if iteration < num_iterations - 1:
                a = (u_hat[:, None, :, :] * v[:, :, None, :]).sum(dim=-1)
                b += a

        return v

    def squash(self, tensor, dim=-1):
        squared_norm = (tensor ** 2).sum(dim=dim, keepdim=True)
        scale = squared_norm / (1 + squared_norm)
        return scale * tensor / torch.sqrt(squared_norm)

class CapsuleRGCN(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_relations, num_capsules, capsule_size, num_classes):
        super(CapsuleRGCN, self).__init__()

        self.conv1 = RGCNConv(input_dim, hidden_dim, num_relations)
        self.capsule_layer = CapsuleLayer(hidden_dim, output_dim, num_capsules, capsule_size)
        self.linear = nn.Linear(num_capsules * capsule_size, num_classes)

    def forward(self, x, edge_index, edge_type):
        # Perform the first convolutional layer
        x = F.relu(self.conv1(x, edge_index, edge_type))

        # Perform the capsule layer
        capsules = self.capsule_layer(x, edge_index, edge_type)

        # Flatten and transform the capsule tensor to a classification score
        v = capsules.view(capsules.size(0), -1)
        scores = self.linear(v)

        return scores
