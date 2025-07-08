import torch
import torch.nn as nn
from torch.nn import init
import numbers
import torch.nn.functional as F
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
import numpy as np


class graph_constructor(nn.Module):
    def __init__(self, nnodes, k, dim, device, alpha=3, static_feat=None):
        super(graph_constructor, self).__init__()
        self.nnodes = nnodes
        if static_feat is not None:
            xd = static_feat.shape[1]
            self.lin1 = nn.Linear(xd, dim)
            self.lin2 = nn.Linear(xd, dim)
        else:
            self.emb1 = nn.Embedding(nnodes, dim)
            self.emb2 = nn.Embedding(nnodes, dim)
            self.lin1 = nn.Linear(dim,dim)
            self.lin2 = nn.Linear(dim,dim)

        self.device = device
        self.k = k
        self.dim = dim
        self.alpha = alpha
        self.static_feat = static_feat

    def forward(self, idx):
        device = self.emb1.weight.device
        idx = idx.to(device)
        if self.static_feat is None:
            nodevec1 = self.emb1(idx)
            nodevec2 = self.emb2(idx)
        else:
            nodevec1 = self.static_feat[idx,:]
            nodevec2 = nodevec1
        # nodevec1 = nodevec1#.to(torch.bfloat16)
        # nodevec2 = nodevec2#.to(torch.bfloat16)
        # print(type(idx))
        nodevec1 = torch.tanh(self.alpha*self.lin1(nodevec1))    #  M_1 =  tanh(\alpha h^{(1)} W_1 )
        nodevec2 = torch.tanh(self.alpha*self.lin2(nodevec2))    #  M_2 =  tanh(\alpha h^{(2)} W_2 )

        '''
        A = ReLU(tanh(\alpha (M_1*M_2^{T} - M_{2}^{T}*M_1))) 
        '''
        a = torch.mm(nodevec1, nodevec2.transpose(1,0))-torch.mm(nodevec2, nodevec1.transpose(1,0))    
        adj = F.relu(torch.tanh(self.alpha*a))


        s1,t1 = (adj + torch.rand_like(adj)*0.01).topk(self.k,1)


        mask = torch.zeros(idx.size(0), idx.size(0)).to(device)


        mask.fill_(0)#.to(torch.bfloat16) 
        s1 = s1.float() # Convert s1 to float
        
        s1.fill_(1)
        s1 = s1.to(device)
        t1 = t1.to(device)
        mask.scatter_(1,t1,s1.fill_(1))
        adj = adj*mask
        adj = adj
        return adj


    def fullA(self, idx):
        
        if self.static_feat is None:
            nodevec1 = self.emb1(idx)
            nodevec2 = self.emb2(idx)
        else:
            nodevec1 = self.static_feat[idx,:]
            nodevec2 = nodevec1

        nodevec1 = torch.tanh(self.alpha*self.lin1(nodevec1))
        nodevec2 = torch.tanh(self.alpha*self.lin2(nodevec2))

        a = torch.mm(nodevec1, nodevec2.transpose(1,0))-torch.mm(nodevec2, nodevec1.transpose(1,0))
        adj = F.relu(torch.tanh(self.alpha*a))
        return adj
    




class graph_global(nn.Module):
    def __init__(self, nnodes, k, dim, device, alpha=3, static_feat=None):
        super(graph_global, self).__init__()
        self.nnodes = nnodes
        self.A = nn.Parameter(torch.randn(nnodes, nnodes).to(device), requires_grad=True).to(device)
        self.A#.to(torch.bfloat16)

    def forward(self, idx):
        return F.relu(self.A)



class graph_undirected(nn.Module):
    def __init__(self, nnodes, k, dim, device, alpha=3, static_feat=None):
        super(graph_undirected, self).__init__()
        self.nnodes = nnodes
        if static_feat is not None:
            xd = static_feat.shape[1]
            self.lin1 = nn.Linear(xd, dim)
        else:
            self.emb1 = nn.Embedding(nnodes, dim)
            self.lin1 = nn.Linear(dim,dim)

        self.device = device
        self.k = k
        self.dim = dim
        self.alpha = alpha
        self.static_feat = static_feat

    def forward(self, idx):
        if self.static_feat is None:
            nodevec1 = self.emb1(idx)
            nodevec2 = self.emb1(idx)
        else:
            nodevec1 = self.static_feat[idx,:]
            nodevec2 = nodevec1

        nodevec1 = torch.tanh(self.alpha*self.lin1(nodevec1))
        nodevec2 = torch.tanh(self.alpha*self.lin1(nodevec2))

        a = torch.mm(nodevec1, nodevec2.transpose(1,0))
        adj = F.relu(torch.tanh(self.alpha*a))
        mask = torch.zeros(idx.size(0), idx.size(0)).to(self.device)
        mask.fill_(0)#.to(torch.bfloat16)
        s1,t1 = adj.topk(self.k,1)
        mask.scatter_(1,t1,s1.fill_(1))
        adj = adj*mask
        return adj



class graph_directed(nn.Module):
    def __init__(self, nnodes, k, dim, device, alpha=3, static_feat=None):
        super(graph_directed, self).__init__()
        self.nnodes = nnodes
        if static_feat is not None:
            xd = static_feat.shape[1]
            self.lin1 = nn.Linear(xd, dim)
            self.lin2 = nn.Linear(xd, dim)
        else:
            self.emb1 = nn.Embedding(nnodes, dim)
            self.emb2 = nn.Embedding(nnodes, dim)
            self.lin1 = nn.Linear(dim,dim)
            self.lin2 = nn.Linear(dim,dim)

        self.device = device
        self.k = k
        self.dim = dim
        self.alpha = alpha
        self.static_feat = static_feat

    def forward(self, idx):
        if self.static_feat is None:
            nodevec1 = self.emb1(idx)
            nodevec2 = self.emb2(idx)
        else:
            nodevec1 = self.static_feat[idx,:]
            nodevec2 = nodevec1

        nodevec1 = torch.tanh(self.alpha*self.lin1(nodevec1))
        nodevec2 = torch.tanh(self.alpha*self.lin2(nodevec2))

        a = torch.mm(nodevec1, nodevec2.transpose(1,0))
        adj = F.relu(torch.tanh(self.alpha*a))
        mask = torch.zeros(idx.size(0), idx.size(0)).to(self.device)#.to(torch.bfloat16)
        mask.fill_(0)
        s1,t1 = adj.topk(self.k,1)
        mask.scatter_(1,t1,s1.fill_(1))
        adj = adj*mask
        return adj





class directedhypergraph_constructor:
    def __init__(self, num_hyperedges, nnodes, metric='knn', max_nodes_per_hyperedge=None, similarity_threshold=0.5, num_neighbors=3):
        self.num_hyperedges = num_hyperedges
        self.nnodes = nnodes
        self.metric = metric
        self.max_nodes_per_hyperedge = max_nodes_per_hyperedge
        self.similarity_threshold = similarity_threshold
        self.num_neighbors = num_neighbors

    def build_hypergraph(self, transformed_features, device):
        if self.metric == 'cluster':
            kmeans = KMeans(n_clusters=self.num_hyperedges).fit(transformed_features.cpu().detach().numpy())
            cluster_labels = torch.tensor(kmeans.labels_, dtype=torch.long, device=device)

            # Initialize the incidence matrix
            H = torch.zeros((self.nnodes, self.num_hyperedges), device=device)

            if self.max_nodes_per_hyperedge is not None:
                for i in range(self.nnodes):
                    hyperedge_idx = cluster_labels[i]
                    if torch.sum(H[:, hyperedge_idx].abs()) < self.max_nodes_per_hyperedge:
                        H[i, hyperedge_idx] = 1  # Assign as head node
                    else:
                        H[i, hyperedge_idx] = -1  # Assign as tail node
            else:
                for i in range(self.nnodes):
                    hyperedge_idx = cluster_labels[i]
                    H[i, hyperedge_idx] = 1  # Assign as head node

        elif self.metric == 'cosine':
            sim_matrix = F.cosine_similarity(transformed_features.unsqueeze(1), transformed_features.unsqueeze(0), dim=-1)
            sim_matrix[sim_matrix < self.similarity_threshold] = 0
            _, topk_indices = torch.topk(sim_matrix, self.num_hyperedges, dim=1)

            H = torch.zeros((self.nnodes, self.num_hyperedges), device=device)

            for i, indices in enumerate(topk_indices):
                # Assign higher similarity nodes as head nodes (1) and lower as tail nodes (-1)
                # This is a simplistic approach; adjust based on specific criteria
                H[indices[:self.num_hyperedges//2], i] = 1
                if len(indices) > self.num_hyperedges//2:
                    H[indices[self.num_hyperedges//2:], i] = -1

        elif self.metric == 'knn':
            num_neighbors = self.num_neighbors
            transformed_features = transformed_features.to(torch.float32)
            nn = NearestNeighbors(n_neighbors=num_neighbors, metric='euclidean').fit(transformed_features.cpu().detach().numpy())
            distances, indices = nn.kneighbors(transformed_features.cpu().detach().numpy())

            hyperedges = set()
            for i in range(self.nnodes):
                hyperedge = frozenset(indices[i])
                hyperedges.add(hyperedge)

            unique_hyperedges = list(hyperedges)
            H = torch.zeros((self.nnodes, len(unique_hyperedges)), device=device)

            for idx, hyperedge in enumerate(unique_hyperedges):
                # Assign the closest node as head (1) and others as tail (-1)
                nodes = list(hyperedge)
                closest_node = nodes[0]  # Assuming the first node is the closest
                H[closest_node, idx] = 1
                for node in nodes[1:]:
                    H[node, idx] = -1

        # Convert to bfloat16 if needed
        H = H.to(torch.bfloat16)

        return H
    


#---------------------------------------------------------------------------
#%% Constructing Hypergraph
#---------------------------------------------------------------------------
class hgypergraph_constructor(nn.Module):
    '''
    This class used for constructing hypergraph from the x_enc ,
    Design objective: Constructing Adaptive Hypergraph Structure for integrating Coupling relationship 
    between Multiple time series variables
    Param:
        input:  
    '''
    def __init__(self, nnodes, num_hyperedges,  dim, device, scale_hyperedges = 10, alpha=3, static_feat=None,metric='knn' , max_nodes_per_hyperedge=None,similarity_threshold=1):
        super(hgypergraph_constructor, self).__init__()
        self.nnodes = nnodes
        self.num_hyperedges = num_hyperedges
        self.device = device
        self.dim = dim  # input dimension
        self.alpha = alpha  
        self.static_feat = static_feat
        self.max_nodes_per_hyperedge = max_nodes_per_hyperedge 
        self.metric = metric
        self.similarity_threshold = similarity_threshold
        self.scale_hyperedges = scale_hyperedges
        # self.num_neighbors = nn.Parameter(torch.tensor(10)).to(torch.bfloat16)


        if static_feat is not None:
            xd = static_feat.shape[1]
            self.lin1 = nn.Linear(xd, dim)
        else:
            self.emb1 = nn.Embedding(nnodes, dim)  
            self.lin1 = nn.Linear(dim,dim)   


    def forward(self, idx):
        device = self.emb1.weight.device
        idx = idx.to(device)


        if self.static_feat is None:
            node_features = self.emb1(idx)
        else:
            node_features = self.static_feat[idx, :]

        # node_features = node_features.to(torch.bfloat16)
        # print(type(idx))
        transformed_features = torch.tanh(self.alpha * self.lin1(node_features))

        # Generate hyperedges using a clustering algorithm or any other method

        if self.metric == 'cluster':
            kmeans = KMeans(n_clusters=self.num_hyperedges).fit(transformed_features.cpu().detach().numpy())
            cluster_labels = torch.tensor(kmeans.labels_,dtype=torch.long,device = device)
            # Construct the hyperedge-node incidence matrix H
            H = torch.zeros((self.num_hyperedges, self.nnodes),device = device)
            # Optionally apply a mask to limit the number od nodes per hyperedges
            if self.max_nodes_per_hyperedge is not None:
                for i in range(self.nnodes):
                    hyperedge_idx = cluster_labels[i]
                    if torch.sum(H[hyperedge_idx]) < self.max_nodes_per_hyperedge:
                            H[hyperedge_idx,i] = 1
            else: 
                for i in range(self.nnodes):
                    H[cluster_labels[i],i] = 1
        elif self.metric =='cosine':
            # Generate hyperedges based on similarity
            sim_matrix = F.cosine_similarity(transformed_features.unsqueeze(1), transformed_features.unsqueeze(0),dim=-1)
            sim_matrix[sim_matrix < self.similarity_threshold] = 0
            _, topk_indices = torch.topk(sim_matrix, self.num_hyperedges, dim=1)
            H = torch.zeros((self.num_hyperedges, self.nnodes),device=device)
            for i, indices in enumerate(topk_indices):
                H[torch.arrage(self.num_hyperedges), indices] = 1
        

        elif self.metric == 'knn':
            # Generate hyperedges based on k-nearest neighbors method
            # num_neighbors_clamped = torch.clamp(self.num_neighbors, min=3, max=np.sqrt(self.nnodes))
            # num_neighbors = torch.round(num_neighbors_clamped)

            num_neighbors = self.scale_hyperedges    #int(np.sqrt(self.nnodes))

            transformed_features = transformed_features.to(torch.float32)
            nn = NearestNeighbors(n_neighbors=num_neighbors, metric='euclidean').fit(transformed_features.cpu().detach().numpy())
            distances, indices = nn.kneighbors(transformed_features.cpu().detach().numpy())

            hyperedges = set()
            for i in range(self.nnodes):
                # Include the node itself and its k nearest neighbors as a hyperedge
                # Use frozenset to ensure uniqueness of hyperedges (sets are not hashable)
                hyperedge = frozenset(indices[i])
                hyperedges.add(hyperedge)


            # Convert the set of unique hyperedges back to a list
            unique_hyperedges = list(hyperedges)

            # If we have more hyperedges than needed, randomly select num_hyperedges of them
            # if len(unique_hyperedges) > self.num_hyperedges:
            #     unique_hyperedges = random.sample(unique_hyperedges, self.num_hyperedges)

            # Initialize the hyperedge incidence matrix H

            H = torch.zeros((len(unique_hyperedges), self.nnodes), device=device)

            # Populate the hyperedge incidence matrix
            for idx, hyperedge in enumerate(unique_hyperedges):
                for node in hyperedge:
                    H[idx, node] = 1
        H = H.t()
        H = H.to(torch.bfloat16)
        return H


    def get_hypergraph_matrix(self, idx):
        """
        This method generates the hypergraph structure based on the node features.
        It constructs the hyperedge-node incidence matrix H.
        """
        device = self.emb1.weight.device
        idx = idx.to(device)

        if self.static_feat is None:
            node_features = self.emb1(idx)
        else:
            node_features = self.static_feat[idx, :]

        # Transform node features
        transformed_features = torch.tanh(self.alpha * self.lin1(node_features))

        # Generate hyperedges using a clustering algorithm or any other method
        kmeans = KMeans(n_clusters=self.num_hyperedges).fit(transformed_features.cpu().detach().numpy())
        cluster_labels = torch.tensor(kmeans.labels_, dtype=torch.long)

        # Construct the hyperedge-node incidence matrix H
        H = torch.zeros((self.num_hyperedges, self.nnodes))
        
        # Optionally apply a mask to limit the number of nodes per hyperedge
        if self.max_nodes_per_hyperedge is not None:
            for i in range(self.nnodes):
                hyperedge_idx = cluster_labels[i]
                if torch.sum(H[hyperedge_idx]) < self.max_nodes_per_hyperedge:
                    H[hyperedge_idx, i] = 1
        else:
            for i in range(self.nnodes):
                H[cluster_labels[i], i] = 1

        return H


class hypergraph_Global(nn.Module):
    def __init__(self, nnodes, num_hyperedges, dim, device, alpha=3, static_feat=None):
        super(hypergraph_Global, self).__init__()
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
    


class hypergraph_Undirected(nn.Module):
    def __init__(self, nnodes, num_hyperedges, dim, device, alpha=3, static_feat=None, metric='knn',similarity_threshold=1):
        super(hypergraph_Undirected, self).__init__()
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


        elif self.metric =='knn':
            # Generate hyperedges based on k-nearest neighbors method
            num_neighbors = 4
            nn = NearestNeighbors(n_neighbors=num_neighbors, metric='euclidean').fit(nodevec1.cpu().detach().numpy())
            distances, indices = nn.kneighbors(nodevec1.cpu().detach().numpy())

            hyperedges = set()
            for i in range(self.nnodes):
                # Include the node itself and its k nearest neighbors as a hyperedge
                # Use frozenset to ensure uniqueness of hyperedges (sets are not hashable)
                hyperedge = frozenset(indices[i])
                hyperedges.add(hyperedge)


            # Convert the set of unique hyperedges back to a list
            unique_hyperedges = list(hyperedges)

            # If we have more hyperedges than needed, randomly select num_hyperedges of them
            # if len(unique_hyperedges) > self.num_hyperedges:
            #     unique_hyperedges = random.sample(unique_hyperedges, self.num_hyperedges)

            # Initialize the hyperedge incidence matrix H

            H = torch.zeros((len(unique_hyperedges), self.nnodes), device=device)

            # Populate the hyperedge incidence matrix
            for idx, hyperedge in enumerate(unique_hyperedges):
                for node in hyperedge:
                    H[idx, node] = 1



        else:
            raise ValueError("Hypergraph Construction Method is not Defined!")

        return H


