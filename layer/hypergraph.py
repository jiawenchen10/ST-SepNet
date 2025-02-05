import torch
import torch.nn as nn
from sklearn.cluster import KMeans
import torch.nn.functional as F


class Hypergraph_Global(nn.Module):
    def __init__(self, nnodes, num_hyperedges, dim, device, alpha=3, static_feat=None):
        super(Hypergraph_Global, self).__init__()
        self.nnodes = nnodes
        self.num_hyperedges = num_hyperedges
        self.device = device
        self.alpha = alpha
        self.static_feat = static_feat

        if static_feat is not None:
            xd = static_feat.shape[1]
            self.lin1 = nn.Linear(xd, dim)
        else:
            self.emb1 = nn.Embedding(nnodes, dim)
            self.lin1 = nn.Linear(dim, dim)

    def forward(self, idx):
        device = self.emb1.weight.device
        idx = idx.to(device)


        if self.static_feat is None:
            node_features = self.emb1(idx)
        else:
            node_features = self.static_feat[idx, :]

        # Transform node features
        transformed_features = torch.tanh(self.alpha * self.lin1(node_features))

        # Generate hyperedges using a clustering algorithm
        kmeans = KMeans(n_clusters=self.num_hyperedges).fit(transformed_features.cpu().detach().numpy())
        cluster_labels = torch.tensor(kmeans.labels_, dtype=torch.long, device = device)

        # Construct the hyperedge-node incidence matrix H
        H = torch.zeros((self.num_hyperedges, self.nnodes), device = device)
        for i in range(self.nnodes):
            H[cluster_labels[i], i] = 1

        return H
    


class Hypergraph_Undirected(nn.Module):
    def __init__(self, nnodes, num_hyperedges, dim, device, alpha=3, static_feat=None, metric='cosine',similarity_threshold=1):
        super(Hypergraph_Undirected, self).__init__()
        self.nnodes = nnodes
        self.num_hyperedges = num_hyperedges
        self.device = device
        self.alpha = alpha
        self.static_feat = static_feat
        self.metric = metric
        self.max_nodes_per_hyperedge = nnodes
        self.similarity_threshold = similarity_threshold


        if static_feat is not None:
            xd = static_feat.shape[1]
            self.lin1 = nn.Linear(xd, dim)
        else:
            self.emb1 = nn.Embedding(nnodes, dim)
            self.lin1 = nn.Linear(dim, dim)

    def forward(self, idx):
        device = self.emb1.weight.device
        idx = idx.to(device)

        if self.static_feat is None:
            nodevec1 = self.emb1(idx)
        else:
            nodevec1 = self.static_feat[idx, :]

        nodevec1 = torch.tanh(self.alpha * self.lin1(nodevec1))

        # Costucting HyperGraph between nodes
        if self.metric == 'cosine':
            # Generate hyperedges based on similarity
            sim_matrix = F.cosine_similarity(nodevec1.unsqueeze(1), nodevec1.unsqueeze(0), dim=-1)
            sim_matrix[sim_matrix < self.similarity_threshold] = 0
            _, topk_indices = torch.topk(sim_matrix, self.num_hyperedges, dim=1)
            H = torch.zeros((self.num_hyperedges, self.nnodes), device = device)
            for i, indices in enumerate(topk_indices):
                H[torch.arange(self.num_hyperedges), indices] = 1

        elif self.metric =='cluster':
            # Generate hyperedges based on k-cluster method 
            kmeans = KMeans(n_clusters= self.num_hyperedges).fit(nodevec1.cpu().detach().numpy())
            cluster_labels = torch.tensor(kmeans.labels_, dtype = torch.long, device=device)
            H = torch.zeros((self.num_hyperedges, self.nnodes), device = device)
            for i in range(self.nnodes):
                H[cluster_labels[i],i] = 1
    
        else:
            raise ValueError("Hypergraph Construction Method is not Defined!")

        return H
    


# Define the hypergraph parameters
nnodes = 10  
num_hyperedges = 3  
dim = 5  # features dimension of node 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  #
alpha = 3  # param of activation function
static_feat = None  # 

# node index 
idx = torch.arange(nnodes, dtype=torch.long, device=device)

# initialize the hypergraph
hypergraph_global = Hypergraph_Global(nnodes, num_hyperedges, dim, device, alpha, static_feat)
H_global = hypergraph_global(idx)
print("Global Hyperedge-Node Incidence Matrix H:")
print(H_global.T)



metric='cosine'
hypergraph_undirected = Hypergraph_Undirected(nnodes, num_hyperedges, dim, device, alpha, static_feat, metric )
H_undirected = hypergraph_undirected(idx)
print("\nUndirected Hyperedge-Node Incidence Matrix H:")
print(H_undirected.T)

    

