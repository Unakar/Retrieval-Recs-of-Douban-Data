import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import random
truncate_len = 30


class ID_Mapper:
    def __init__(self, id_list):
        self.unique_ids = sorted(set(id_list))
    
        # 创建将原始ID映射到连续索引的字典
        self.Idx_map = {id: idx for idx, id in enumerate(self.unique_ids, start = 1)}
    
        # 创建将连续索引映射回原始ID的字典
        self.map_back = {idx: id for id, idx in self.Idx_map.items()}
    
    def __len__(self):
        return len(self.unique_ids)

    def get_idx_map(self, item):
        return self.Idx_map[item]
    
    def get_map_back(self, idx):
        return self.map_back[idx]
    

class Rating_Dataloader(Dataset):
    def __init__(
            self,
            douban_type, # movie or book
            data,
            user_mapper,
            item_mapper,
            tag_embed_dict,
    ):
        if douban_type not in ['movie', 'book']:
            raise ValueError('douban_type must be "movie" or "book"')
        self.douban_type = douban_type
        
        self.user_mapper = user_mapper
        self.item_mapper = item_mapper
        self.tag_embed_dict = tag_embed_dict

        self.item_id = data[self.douban_type].values
        self.user_id = data['User'].values
        self.rating = data['Rating'].astype("float32").values

    def __len__(self):
        return len(self.rating)
    
    def __getitem__(self, index):
        item = self.item_id[index]
        user_index = self.user_mapper.get_idx_map(self.user_id[index])
        item_index = self.item_mapper.get_idx_map(item)
        rating = self.rating[index]
        text_embed = self.tag_embed_dict.get(item)
        return user_index, item_index, rating, text_embed
    

class Contact_Dataloader(Dataset):
    def __init__(
            self,
            douban_type, # movie or book
            data,
            user_mapper,
            item_mapper,
            user_item_dict,
            user_user_dict,
            user_user_item_dict,
            item_user_dict,
    ):
        if douban_type not in ['movie', 'book']:
            raise ValueError('douban_type must be "movie" or "book"')
        self.douban_type = douban_type
        
        self.user_mapper = user_mapper
        self.item_mapper = item_mapper

        self.user_item_dict = user_item_dict
        self.user_user_dict = user_user_dict
        self.user_user_item_dict = user_user_item_dict
        self.item_user_dict = item_user_dict

        self.item_id = data[self.douban_type].values
        self.user_id = data['User'].values
        self.rating = data['Rating'].astype("float32").values

    def __len__(self):
        return len(self.rating)
    
    def __getitem__(self, index):
        item = self.item_id[index]
        user_index = self.user_mapper.get_idx_map(self.user_id[index])
        item_index = self.item_mapper.get_idx_map(item)
        rating = self.rating[index]

        # user-item
        user_items = self.user_item_dict.get(self.user_id[index], [])
        
        # user-user
        user_users = self.user_user_dict.get(self.user_id[index], [])
        
        # user-user-item
        user_users_items = []
        for uu in user_users:
            user_users_items.append(self.user_item_dict.get(uu, []))
        
        # item-user
        item_users = self.item_user_dict.get(item, [])
        
        return (user_index, item_index, rating, user_items, user_users, user_users_items, item_users)
 
    

def collate_fn(batch_data):
    """This function will be used to pad the graph to max length in the batch
       It will be used in the Dataloader
    """
    uids, iids, labels = [], [], []
    u_items, u_users, u_users_items, i_users = [], [], [], []
    u_items_len, u_users_len, i_users_len = [], [], []

    for data, u_items_u, u_users_u, u_users_items_u, i_users_i in batch_data:

        (uid, iid, label) = data
        uids.append(uid)
        iids.append(iid)
        labels.append(label)

        # user-items    
        if len(u_items_u) <= truncate_len:
            u_items.append(u_items_u)
        else:
            u_items.append(random.sample(u_items_u, truncate_len))
        u_items_len.append(min(len(u_items_u), truncate_len))
        
        # user-users and user-users-items
        if len(u_users_u) <= truncate_len:
            u_users.append(u_users_u)
            u_u_items = [] 
            for uui in u_users_items_u:
                if len(uui) < truncate_len:
                    u_u_items.append(uui)
                else:
                    u_u_items.append(random.sample(uui, truncate_len))
            u_users_items.append(u_u_items)
        else:
            sample_index = random.sample(list(range(len(u_users_u))), truncate_len)
            u_users.append([u_users_u[si] for si in sample_index])

            u_users_items_u_tr = [u_users_items_u[si] for si in sample_index]
            u_u_items = [] 
            for uui in u_users_items_u_tr:
                if len(uui) < truncate_len:
                    u_u_items.append(uui)
                else:
                    u_u_items.append(random.sample(uui, truncate_len))
            u_users_items.append(u_u_items)

        u_users_len.append(min(len(u_users_u), truncate_len))	

        # item-users
        if len(i_users_i) <= truncate_len:
            i_users.append(i_users_i)
        else:
            i_users.append(random.sample(i_users_i, truncate_len))
        i_users_len.append(min(len(i_users_i), truncate_len))

    batch_size = len(batch_data)

    # padding
    u_items_maxlen = max(u_items_len)
    u_users_maxlen = max(u_users_len)
    i_users_maxlen = max(i_users_len)
    
    u_item_pad = torch.zeros([batch_size, u_items_maxlen, 2], dtype=torch.long)
    for i, ui in enumerate(u_items):
        u_item_pad[i, :len(ui), :] = torch.LongTensor(ui)
    
    u_user_pad = torch.zeros([batch_size, u_users_maxlen], dtype=torch.long)
    for i, uu in enumerate(u_users):
        u_user_pad[i, :len(uu)] = torch.LongTensor(uu)
    
    u_user_item_pad = torch.zeros([batch_size, u_users_maxlen, u_items_maxlen, 2], dtype=torch.long)
    for i, uu_items in enumerate(u_users_items):
        for j, ui in enumerate(uu_items):
            u_user_item_pad[i, j, :len(ui), :] = torch.LongTensor(ui)

    i_user_pad = torch.zeros([batch_size, i_users_maxlen, 2], dtype=torch.long)
    for i, iu in enumerate(i_users):
        i_user_pad[i, :len(iu), :] = torch.LongTensor(iu)

    return torch.LongTensor(uids), torch.LongTensor(iids), torch.FloatTensor(labels), \
            u_item_pad, u_user_pad, u_user_item_pad, i_user_pad