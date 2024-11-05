import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MaskedKLDivLoss(nn.Module):
    def __init__(self):
        super(MaskedKLDivLoss, self).__init__()
        self.loss = nn.KLDivLoss(reduction='sum')

    def forward(self, log_pred, target, mask):
        mask_ = mask.view(-1, 1)
        loss = self.loss(log_pred * mask_, target * mask_) / torch.sum(mask)   
        return loss


class MaskedNLLLoss(nn.Module):
    def __init__(self, weight=None):
        super(MaskedNLLLoss, self).__init__()
        self.weight = weight
        self.loss = nn.NLLLoss(weight=weight, reduction='sum')

    def forward(self, pred, target, mask):
        mask_ = mask.view(-1, 1)
        if type(self.weight) == type(None):
            loss = self.loss(pred * mask_, target) / torch.sum(mask)
        else:
            loss = self.loss(pred * mask_, target) \
                   / torch.sum(self.weight[target] * mask_.squeeze())  
        return loss

def gelu(x):
    return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))

class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.actv = gelu
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)

    def forward(self, x):
        inter = self.dropout_1(self.actv(self.w_1(self.layer_norm(x))))
        output = self.dropout_2(self.w_2(inter))
        return output + x


class MultiHeadedAttention(nn.Module):
    def __init__(self, head_count, model_dim, dropout=0.1):
        assert model_dim % head_count == 0
        self.dim_per_head = model_dim // head_count
        self.model_dim = model_dim

        super(MultiHeadedAttention, self).__init__()
        self.head_count = head_count

        self.linear_k = nn.Linear(model_dim, head_count * self.dim_per_head)
        self.linear_v = nn.Linear(model_dim, head_count * self.dim_per_head)
        self.linear_q = nn.Linear(model_dim, head_count * self.dim_per_head)
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(model_dim, model_dim)

    def forward(self, key, value, query, mask=None):
        batch_size = key.size(0)
        dim_per_head = self.dim_per_head
        head_count = self.head_count

        def shape(x):
            """  projection """
            return x.view(batch_size, -1, head_count, dim_per_head).transpose(1, 2)

        def unshape(x):
            """  compute context """
            return x.transpose(1, 2).contiguous() \
                .view(batch_size, -1, head_count * dim_per_head)

        key = self.linear_k(key).view(batch_size, -1, head_count, dim_per_head).transpose(1, 2)
        value = self.linear_v(value).view(batch_size, -1, head_count, dim_per_head).transpose(1, 2)
        query = self.linear_q(query).view(batch_size, -1, head_count, dim_per_head).transpose(1, 2)

        query = query / math.sqrt(dim_per_head)
        scores = torch.matmul(query, key.transpose(2, 3))

        if mask is not None:
            mask = mask.unsqueeze(1).expand_as(scores)
            scores = scores.masked_fill(mask, -1e10)

        attn = self.softmax(scores)
        drop_attn = self.dropout(attn)
        context = torch.matmul(drop_attn, value).transpose(1, 2).\
                    contiguous().view(batch_size, -1, head_count * dim_per_head)
        output = self.linear(context)
        return output


class PositionalEncoding(nn.Module):
    def __init__(self, dim, max_len=512):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp((torch.arange(0, dim, 2, dtype=torch.float) *
                              -(math.log(10000.0) / dim)))
        pe[:, 0::2] = torch.sin(position.float() * div_term)
        pe[:, 1::2] = torch.cos(position.float() * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    def forward(self, x, speaker_emb):
        L = x.size(1)
        pos_emb = self.pe[:, :L]
        x = x + pos_emb + speaker_emb
        return x


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, heads, d_ff, dropout):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = MultiHeadedAttention(
            heads, d_model, dropout=dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, iter, inputs_a, inputs_b, mask):
        if inputs_a.equal(inputs_b):
            if (iter != 0):
                inputs_b = self.layer_norm(inputs_b)
            else:
                inputs_b = inputs_b

            mask = mask.unsqueeze(1)
            context = self.self_attn(inputs_b, inputs_b, inputs_b, mask=mask)
        else:
            if (iter != 0):
                inputs_b = self.layer_norm(inputs_b)
            else:
                inputs_b = inputs_b

            mask = mask.unsqueeze(1)
            context = self.self_attn(inputs_a, inputs_a, inputs_b, mask=mask)
        
        out = self.dropout(context) + inputs_b
        return self.feed_forward(out)


class TransformerEncoder(nn.Module):
    def __init__(self, d_model, d_ff, heads, layers, dropout=0.1):
        super(TransformerEncoder, self).__init__()
        self.d_model = d_model
        self.layers = layers
        self.pos_emb = PositionalEncoding(d_model)
        self.transformer_inter = nn.ModuleList(
            [TransformerEncoderLayer(d_model, heads, d_ff, dropout)
             for _ in range(layers)])
        self.dropout = nn.Dropout(dropout)

    def forward(self, x_a, x_b, mask, speaker_emb):
        if x_a.equal(x_b):
            x_b = self.pos_emb(x_b, speaker_emb)
            x_b = self.dropout(x_b)
            for i in range(self.layers):
                x_b = self.transformer_inter[i](i, x_b, x_b, mask.eq(0))
        else:
            x_a = self.pos_emb(x_a, speaker_emb)
            x_a = self.dropout(x_a)
            x_b = self.pos_emb(x_b, speaker_emb)
            x_b = self.dropout(x_b)
            for i in range(self.layers):
                x_b = self.transformer_inter[i](i, x_a, x_b, mask.eq(0))
        return x_b


class Unimodal_GatedFusion(nn.Module):
    def __init__(self, hidden_size, dataset):
        super(Unimodal_GatedFusion, self).__init__()
        self.fc = nn.Linear(hidden_size, hidden_size, bias=False)
        if dataset == 'MELD':
            self.fc.weight.data.copy_(torch.eye(hidden_size, hidden_size))
            self.fc.weight.requires_grad = False

    def forward(self, a):
        z = torch.sigmoid(self.fc(a))
        final_rep = z * a
        return final_rep

class Multimodal_GatedFusion(nn.Module):
    def __init__(self, hidden_size):
        super(Multimodal_GatedFusion, self).__init__()
        self.fc = nn.Linear(hidden_size, hidden_size, bias=False)
        self.softmax = nn.Softmax(dim=-2)

    def forward(self, a, b, c):
        a_new = a.unsqueeze(-2)
        b_new = b.unsqueeze(-2)
        c_new = c.unsqueeze(-2)
        utters = torch.cat([a_new, b_new, c_new], dim=-2)
        utters_fc = torch.cat([self.fc(a).unsqueeze(-2), self.fc(b).unsqueeze(-2), self.fc(c).unsqueeze(-2)], dim=-2)
        utters_softmax = self.softmax(utters_fc)
        utters_three_model = utters_softmax * utters
        final_rep = torch.sum(utters_three_model, dim=-2, keepdim=False)
        return final_rep

class Transformer_Based_Model(nn.Module):
    def __init__(self, dataset, temp, D_text1, D_text2, D_text3, n_head,
                 n_classes, hidden_dim, n_speakers, dropout):
        super(Transformer_Based_Model, self).__init__()
        self.temp = temp
        self.n_classes = n_classes
        self.n_speakers = n_speakers
        if self.n_speakers == 2:
            padding_idx = 2
        if self.n_speakers == 9:
            padding_idx = 9
        self.speaker_embeddings = nn.Embedding(n_speakers+1, hidden_dim, padding_idx)

        self.hidden_size = 128

        self.rnn = nn.LSTM(input_size=1024, hidden_size=128, num_layers=2,
                               bidirectional=True, dropout=0.2)
        self.rnn_parties = nn.LSTM(input_size=1024, hidden_size=128, num_layers=2,
                                       bidirectional=True, dropout=0.2)

        # Temporal convolutional layers
        self.textf_1_input = nn.Conv1d(D_text1, hidden_dim, kernel_size=1, padding=0, bias=False)
        self.textf_3_input = nn.Conv1d(D_text3, hidden_dim, kernel_size=1, padding=0, bias=False)
        self.textf_2_input = nn.Conv1d(D_text2, hidden_dim, kernel_size=1, padding=0, bias=False)
        
        # Intra- and Inter-modal Transformers
        self.t1_t1 = TransformerEncoder(d_model=hidden_dim, d_ff=hidden_dim, heads=n_head, layers=1, dropout=dropout)
        self.t3_t1 = TransformerEncoder(d_model=hidden_dim, d_ff=hidden_dim, heads=n_head, layers=1, dropout=dropout)
        self.t2_t1 = TransformerEncoder(d_model=hidden_dim, d_ff=hidden_dim, heads=n_head, layers=1, dropout=dropout)

        self.t3_t3 = TransformerEncoder(d_model=hidden_dim, d_ff=hidden_dim, heads=n_head, layers=1, dropout=dropout)
        self.t1_t3 = TransformerEncoder(d_model=hidden_dim, d_ff=hidden_dim, heads=n_head, layers=1, dropout=dropout)
        self.t2_t3 = TransformerEncoder(d_model=hidden_dim, d_ff=hidden_dim, heads=n_head, layers=1, dropout=dropout)

        self.t2_t2 = TransformerEncoder(d_model=hidden_dim, d_ff=hidden_dim, heads=n_head, layers=1, dropout=dropout)
        self.t1_t2 = TransformerEncoder(d_model=hidden_dim, d_ff=hidden_dim, heads=n_head, layers=1, dropout=dropout)
        self.t3_t2 = TransformerEncoder(d_model=hidden_dim, d_ff=hidden_dim, heads=n_head, layers=1, dropout=dropout)
        
        # Unimodal-level Gated Fusion
        self.t1_t1_gate = Unimodal_GatedFusion(hidden_dim, dataset)
        self.t3_t1_gate = Unimodal_GatedFusion(hidden_dim, dataset)
        self.t2_t1_gate = Unimodal_GatedFusion(hidden_dim, dataset)

        self.t3_t3_gate = Unimodal_GatedFusion(hidden_dim, dataset)
        self.t1_t3_gate = Unimodal_GatedFusion(hidden_dim, dataset)
        self.t2_t3_gate = Unimodal_GatedFusion(hidden_dim, dataset)

        self.t2_t2_gate = Unimodal_GatedFusion(hidden_dim, dataset)
        self.t1_t2_gate = Unimodal_GatedFusion(hidden_dim, dataset)
        self.t3_t2_gate = Unimodal_GatedFusion(hidden_dim, dataset)

        self.features_reduce_t1 = nn.Linear(3 * hidden_dim, hidden_dim)
        self.features_reduce_t3 = nn.Linear(3 * hidden_dim, hidden_dim)
        self.features_reduce_t2 = nn.Linear(3 * hidden_dim, hidden_dim)

        # Multimodal-level Gated Fusion
        self.last_gate = Multimodal_GatedFusion(hidden_dim)

        # Emotion Classifier
        self.t1_output_layer = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_classes)
            )
        self.t3_output_layer = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_classes)
            )
        self.t2_output_layer = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_classes)
            )
        self.all_output_layer = nn.Linear(hidden_dim, n_classes)

    def forward(self, textf_1, textf_2, textf_3, u_mask, qmask, dia_len):
        # t2, t3 = None, None
        # t1 = textf_1
        # # (b,l,h), (b,l,p)
        # U_, qmask_ = textf_1.transpose(0, 1), qmask.transpose(0, 1)
        # U_p_ = torch.zeros(U_.size()[0], U_.size()[1], self.hidden_size * 2).type(textf_1.type())
        # U_parties_ = [torch.zeros_like(U_).type(U_.type()) for _ in range(self.n_speakers)]
        # for b in range(U_.size(0)):
        #     for p in range(len(U_parties_)):
        #         index_i = torch.nonzero(qmask_[b][:, p]).squeeze(-1)
        #         if index_i.size(0) > 0:
        #             U_parties_[p][b][:index_i.size(0)] = U_[b][index_i]
        # E_parties_ = [self.rnn_parties(U_parties_[p].transpose(0, 1))[0].transpose(0, 1) for p in
        #               range(len(U_parties_))]
        #
        # for b in range(U_p_.size(0)):
        #     for p in range(len(U_parties_)):
        #         index_i = torch.nonzero(qmask_[b][:, p]).squeeze(-1)
        #         if index_i.size(0) > 0: U_p_[b][index_i] = E_parties_[p][b][:index_i.size(0)]
        # t2 = U_p_.transpose(0, 1)
        #
        # # (l,b,2*h) [(2*bi,b,h) * 2]
        # t3, hidden = self.rnn(textf_1)

        spk_idx = torch.argmax(qmask, -1)
        origin_spk_idx = spk_idx
        if self.n_speakers == 2:
            for i, x in enumerate(dia_len):
                spk_idx[i, x:] = (2*torch.ones(origin_spk_idx[i].size(0)-x)).int().cuda()
        if self.n_speakers == 9:
            for i, x in enumerate(dia_len):
                spk_idx[i, x:] = (9*torch.ones(origin_spk_idx[i].size(0)-x)).int().cuda()
        spk_embeddings = self.speaker_embeddings(spk_idx)

        # Temporal convolutional layers
        textf_1 = self.textf_1_input(textf_1.permute(1, 2, 0)).transpose(1, 2)
        textf_3 = self.textf_3_input(textf_3.permute(1, 2, 0)).transpose(1, 2)
        textf_2 = self.textf_2_input(textf_2.permute(1, 2, 0)).transpose(1, 2)

        # Intra- and Inter-modal Transformers
        t1_t1_transformer_out = self.t1_t1(textf_1, textf_1, u_mask, spk_embeddings)
        t3_t1_transformer_out = self.t3_t1(textf_3, textf_1, u_mask, spk_embeddings)
        t2_t1_transformer_out = self.t2_t1(textf_2, textf_1, u_mask, spk_embeddings)

        t3_t3_transformer_out = self.t3_t3(textf_3, textf_3, u_mask, spk_embeddings)
        t1_t3_transformer_out = self.t1_t3(textf_1, textf_3, u_mask, spk_embeddings)
        t2_t3_transformer_out = self.t2_t3(textf_2, textf_3, u_mask, spk_embeddings)

        t2_t2_transformer_out = self.t2_t2(textf_2, textf_2, u_mask, spk_embeddings)
        t1_t2_transformer_out = self.t1_t2(textf_1, textf_2, u_mask, spk_embeddings)
        t3_t2_transformer_out = self.t3_t2(textf_3, textf_2, u_mask, spk_embeddings)

        # Unimodal-level Gated Fusion
        t1_t1_transformer_out = self.t1_t1_gate(t1_t1_transformer_out)
        t3_t1_transformer_out = self.t3_t1_gate(t3_t1_transformer_out)
        t2_t1_transformer_out = self.t2_t1_gate(t2_t1_transformer_out)

        t3_t3_transformer_out = self.t3_t3_gate(t3_t3_transformer_out)
        t1_t3_transformer_out = self.t1_t3_gate(t1_t3_transformer_out)
        t2_t3_transformer_out = self.t2_t3_gate(t2_t3_transformer_out)

        t2_t2_transformer_out = self.t2_t2_gate(t2_t2_transformer_out)
        t1_t2_transformer_out = self.t1_t2_gate(t1_t2_transformer_out)
        t3_t2_transformer_out = self.t3_t2_gate(t3_t2_transformer_out)

        t1_transformer_out = self.features_reduce_t1(torch.cat([t1_t1_transformer_out, t3_t1_transformer_out, t2_t1_transformer_out], dim=-1))
        t3_transformer_out = self.features_reduce_t3(torch.cat([t3_t3_transformer_out, t1_t3_transformer_out, t2_t3_transformer_out], dim=-1))
        t2_transformer_out = self.features_reduce_t2(torch.cat([t2_t2_transformer_out, t1_t2_transformer_out, t3_t2_transformer_out], dim=-1))

        # Multimodal-level Gated Fusion
        all_transformer_out = self.last_gate(t1_transformer_out, t3_transformer_out, t2_transformer_out)

        # Emotion Classifier
        t1_final_out = self.t1_output_layer(t1_transformer_out)
        t3_final_out = self.t3_output_layer(t3_transformer_out)
        t2_final_out = self.t2_output_layer(t2_transformer_out)
        all_final_out = self.all_output_layer(all_transformer_out)

        t1_log_prob = F.log_softmax(t1_final_out, 2)
        t3_log_prob = F.log_softmax(t3_final_out, 2)
        t2_log_prob = F.log_softmax(t2_final_out, 2)

        all_log_prob = F.log_softmax(all_final_out, 2)
        all_prob = F.softmax(all_final_out, 2)

        kl_t1_log_prob = F.log_softmax(t1_final_out /self.temp, 2)
        kl_t3_log_prob = F.log_softmax(t3_final_out /self.temp, 2)
        kl_t2_log_prob = F.log_softmax(t2_final_out /self.temp, 2)

        kl_all_prob = F.softmax(all_final_out /self.temp, 2)

        return t1_log_prob, t3_log_prob, t2_log_prob, all_log_prob, all_prob, \
               kl_t1_log_prob, kl_t3_log_prob, kl_t2_log_prob, kl_all_prob
