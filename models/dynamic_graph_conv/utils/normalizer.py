import torch
import torch.nn as nn

from utils.cal_adj import remove_nan_inf


class Normalizer(nn.Module):
    def __init__(self):
        super().__init__()

    def _norm(self, graph):
        degree  = torch.sum(graph, dim=2)
        degree  = remove_nan_inf(1 / degree)
        degree  = torch.diag_embed(degree)
        normed_graph = torch.bmm(degree, graph)
        return normed_graph

    def forward(self, adj):
        return [self._norm(_) for _ in adj]

class MultiOrder(nn.Module):
    def __init__(self, order=2):
        super().__init__()
        self.order  = order

    def _multi_order(self, graph):
        graph_ordered = [] # graph_ordered用于存放 1~ks 阶的邻接矩阵
        k_1_order = graph               # 1 order
        mask = torch.eye(graph.shape[1]).to(graph.device)
        mask = 1 - mask
        graph_ordered.append(k_1_order * mask) 
        for k in range(2, self.order+1): 
            k_1_order = torch.matmul(k_1_order, graph)
            graph_ordered.append(k_1_order * mask)
        return graph_ordered

    def forward(self, adj):
        return [self._multi_order(_) for _ in adj]
