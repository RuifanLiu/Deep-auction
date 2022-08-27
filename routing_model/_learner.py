from routing_model.layers import TransformerEncoder, MultiHeadAttention

import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionLearner(nn.Module):
    def __init__(self, cust_feat_size, veh_state_size, model_size = 128,
            layer_count = 3, head_count = 8, ff_size = 512, tanh_xplor = 10, greedy = False):
        r"""
        :param model_size:  Dimension :math:`D` shared by all intermediate layers
        :param layer_count: Number of layers in customers' (graph) Transformer Encoder
        :param head_count:  Number of heads in all Multi-Head Attention layers
        :param ff_size:     Dimension of the feed-forward sublayers in Transformer Encoder
        :param tanh_xplor:  Enable tanh exploration and set its amplitude
        """
        super().__init__()

        self.model_size = model_size
        self.inv_sqrt_d = model_size ** -0.5

        self.tanh_xplor = tanh_xplor

        self.depot_embedding = nn.Linear(cust_feat_size, model_size)
        self.cust_embedding  = nn.Linear(cust_feat_size, model_size)
        self.cust_encoder    = TransformerEncoder(layer_count, head_count, model_size, ff_size)

        self.fleet_attention = MultiHeadAttention(head_count, veh_state_size, model_size)
        self.veh_attention   = MultiHeadAttention(head_count, model_size)
        self.cust_project    = nn.Linear(model_size, model_size)

        self.greedy = greedy


    def _encode_customers(self, customers, mask = None):
        r"""
        :param customers: :math:`N \times L_c \times D_c` tensor containing minibatch of customers' features
        :param mask:      :math:`N \times L_c` tensor containing minibatch of masks
                where :math:`m_{nj} = 1` if customer :math:`j` in sample :math:`n` is hidden (pad or dyn), 0 otherwise
        """
        cust_emb = torch.cat((
            self.depot_embedding(customers[:,0:1,:]),
            self.cust_embedding(customers[:,1:,:])
            ), dim = 1) #.size() = N x L_c x D
        if mask is not None:
            cust_emb[mask] = 0
        self.cust_enc = self.cust_encoder(cust_emb, mask) #.size() = N x L_c x D
        self.fleet_attention.precompute(self.cust_enc)
        self.cust_repr = self.cust_project(self.cust_enc) #.size() = N x L_c x D
        if mask is not None:
            self.cust_repr[mask] = 0


    def _repr_vehicle(self, vehicles, veh_idx, mask):
        r"""
        :param vehicles: :math:`N \times L_v \times D_v` tensor containing minibatch of vehicles' states
        :param veh_idx:  :math:`N \times 1` tensor containing minibatch of indices corresponding to currently acting vehicle
        :param mask:     :math:`N \times L_v \times L_c` tensor containing minibatch of masks
                where :math:`m_{nij} = 1` if vehicle :math:`i` cannot serve customer :math:`j` in sample :math:`n`, 0 otherwise

        :return:         :math:`N \times 1 \times D` tensor containing minibatch of representations for currently acting vehicle
        """
        fleet_repr = self.fleet_attention(vehicles, mask = mask) #.size() = N x L_v x D
        veh_query = fleet_repr.gather(1, veh_idx.unsqueeze(2).expand(-1, -1, self.model_size)) #.size() = N x 1 x D
        return self.veh_attention(veh_query, fleet_repr, fleet_repr) #.size() = N x 1 x D


    def _score_customers(self, veh_repr):
        r"""
        :param veh_repr: :math:`N \times 1 \times D` tensor containing minibatch of representations for currently acting vehicle

        :return:         :math:`N \times 1 \times L_c` tensor containing minibatch of compatibility scores between currently acting vehicle and each customer
        """
        compat = veh_repr.matmul( self.cust_repr.transpose(1, 2) ) #.size() = N x 1 x L_c
        compat *= self.inv_sqrt_d
        if self.tanh_xplor is not None:
            compat = self.tanh_xplor * compat.tanh()
        return compat


    def _get_logp(self, compat, veh_mask):
        r"""
        :param compat:   :math:`N \times 1 \times L_c` tensor containing minibatch of compatibility scores between currently acting vehicle and each customer
        :param veh_mask: :math:`N \times 1 \times L_c` tensor containing minibatch of masks
                where :math:`m_{nj} = 1` if currently acting vehicle cannot serve customer :math:`j` in sample :math:`n`, 0 otherwise

        :return:         :math:`N \times L_c` tensor containing minibatch of log-probabilities for choosing which customer to serve next
        """
        compat[veh_mask] = -float('inf')
        return compat.log_softmax(dim = 2).squeeze(1)


    def step(self, dyna):
        veh_repr = self._repr_vehicle(dyna.vehicles, dyna.cur_veh_idx, dyna.mask)
        compat = self._score_customers(veh_repr)
        logp = self._get_logp(compat, dyna.cur_veh_mask)
        if self.greedy:
            cust_idx = logp.argmax(dim = 1, keepdim = True)
        else:
            cust_idx = logp.exp().multinomial(1)
        return cust_idx, logp.gather(1, cust_idx)


    def forward(self, dyna):
        dyna.reset()
        actions, logps, rewards = [], [], []
        while not dyna.done:
            if dyna.new_customers:
                self._encode_customers(dyna.nodes, dyna.cust_mask)
            cust_idx, logp = self.step(dyna)
            actions.append( (dyna.cur_veh_idx, cust_idx) )
            logps.append( logp )
            rewards.append( dyna.step(cust_idx) )
        return actions, logps, rewards
