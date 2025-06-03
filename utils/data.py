import torch
from torch.utils.data import Dataset
import numpy as np

class VAEData(Dataset):
    def __init__(self, num_item):
        super().__init__()

        self.num_item = num_item

    def __len__(self):
        return self.num_item
    
    def __getitem__(self, idx):
        return idx
    
class GPT2RecBatchTrainAuxData(Dataset):
    def __init__(self, seqs_dict, num_item, max_length=256):
        super().__init__()
        self.max_length = max_length
        self.num_item = num_item

        self.padding_id = 15360

        self.seqs = []
        for _ in seqs_dict:
            self.seqs.append(seqs_dict[_])

        with open('./data/pre_train/all_item_text_embeddings.npy', 'rb') as f:
            item_text_embeddings = np.load(f)
            self.item_text_embeddings = torch.tensor(item_text_embeddings).float()

    def setTokenIdList(self, token_id_list):
        self.token_id_list = token_id_list

    def __len__(self):
        return len(self.seqs)
    
    def encode_aux(self, batch_ids, num_item):
        embed_mask = (batch_ids > -1).long()
        embeds_ids = (batch_ids * embed_mask).clamp_(0, num_item-1)

        batch_embeds = self.item_text_embeddings[embeds_ids]
        batch_embeds = batch_embeds.view(batch_embeds.shape[0] * 4, 768 // 4)

        embed_mask = embed_mask.unsqueeze(-1).repeat(1, 4)
        embed_mask = embed_mask.view(embed_mask.shape[0] * embed_mask.shape[1])

        return batch_embeds, embed_mask.unsqueeze(-1)
    
    def __getitem__(self, idx):
        seq = self.seqs[idx]
        token_list = []
        aux_list = []

        if len(seq) > self.max_length:
            seq = seq[-self.max_length:]

        for item_id in seq:
            token_list.extend(self.token_id_list[item_id])

        # using right padding while training 
        id_list = token_list + [self.padding_id] * (1024 - len(token_list))
        label_list = token_list + [-100] * (1024 - len(token_list))

        aux_list = seq + [-1] * (self.max_length - len(seq))
        aux_list = torch.LongTensor(aux_list)
        batch_aux_embeds, embed_mask = self.encode_aux(aux_list, self.num_item)

        return torch.LongTensor(id_list), torch.LongTensor(label_list), batch_aux_embeds, embed_mask

class GPT2RecBatchEvalAuxData(Dataset):
    def __init__(self, seqs_dict, num_item, max_length=256):
        super().__init__()
        self.max_length = max_length
        self.num_item = num_item

        self.padding_id = 15360

        self.seqs = []
        for _ in seqs_dict:
            self.seqs.append(seqs_dict[_])

    def setTokenIdList(self, token_id_list):
        self.token_id_list = token_id_list

    def setIdMap(self, idMap):
        self.idMap = idMap

    def __len__(self):
        return len(self.seqs)
    
    def __getitem__(self, idx):
        seq = self.seqs[idx]
        token_list = []
        aux_list = []

        if len(seq) > self.max_length:
            seq = seq[-self.max_length:]

        for item_id in seq:
            token_list.extend(self.token_id_list[item_id])

        # using right padding while training 
        id_list = token_list + [self.padding_id] * (1024 - len(token_list))
        label_list = token_list + [-100] * (1024 - len(token_list))

        temp_seq = []
        for item_id in seq:
            temp_seq.append(self.idMap[item_id])
        aux_list = temp_seq + [-1] * (self.max_length - len(temp_seq))

        return torch.LongTensor(id_list), torch.LongTensor(label_list), torch.LongTensor(aux_list)
    
class GPT2RecBatchTestAuxData(Dataset):
    def __init__(self, seqs_dict, num_item, max_length=256):
        super().__init__()
        self.max_length = max_length - 1
        self.num_item = num_item

        self.padding_id = 15360

        self.seq_his = []
        self.seqs = []
        self.targets = []
        for _ in seqs_dict:
            self.seq_his.append(seqs_dict[_][:-1])
            self.seqs.append(seqs_dict[_][:-1])
            self.targets.append(seqs_dict[_][-1])

    def setTokenIdList(self, token_id_list):
        self.token_id_list = token_id_list

    def __len__(self):
        return len(self.seqs)
    
    def __getitem__(self, idx):
        seq = self.seqs[idx]
        token_list = []
        aux_list = []

        if len(seq) > self.max_length:
            seq = seq[-self.max_length:]

        for item_id in seq:
            token_list.extend(self.token_id_list[item_id])

        length = len(token_list)

        # using left padding while inferring
        id_list = [self.padding_id] * (1024 - len(token_list)) + token_list

        aux_list = [-1] * (256 - len(seq)) + seq

        return idx, torch.LongTensor(id_list), self.targets[idx], length, torch.LongTensor(aux_list)

class DummyDataset(Dataset):
    def __init__(self, record_num):
        super().__init__()
        self.record_num = record_num

    def __len__(self):
        return self.record_num
    
    def __getitem__(self, idx):
        return None