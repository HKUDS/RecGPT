import torch

def split_past_key_values(past_key_values):
    """
    将 batch_size 的 past_key_values 拆分成单个样本的 past_key_values。

    参数:
    past_key_values (list of tuples): 每个元素是一个包含两个张量 (key, value) 的元组，
                                       形状为 (batch_size, num_heads, sequence_length, head_dim)。

    返回:
    list of list of tuples: 每个元素是一个单个样本的 past_key_values 列表。
    """
    # 获取 batch_size
    batch_size = past_key_values[0][0].size(0)
    
    # 初始化结果列表
    split_past_key_values = [[] for _ in range(batch_size)]
    
    # 遍历每一层
    for layer in past_key_values:
        key, value = layer
        
        # 拆分 key 和 value
        split_keys = torch.unbind(key, dim=0)
        split_values = torch.unbind(value, dim=0)
        
        # 将拆分后的 key 和 value 添加到对应的样本中
        for i in range(batch_size):
            split_past_key_values[i].append((split_keys[i], split_values[i]))
    
    return split_past_key_values

def combine_past_key_values(single_past_key_values_list):
    """
    将单个样本的 past_key_values 列表恢复成 batch_size 的形式。

    参数:
    single_past_key_values_list (list of list of tuples): 每个元素是一个单个样本的 past_key_values 列表，
                                                          每个 past_key_values 是一个包含两个张量 (key, value) 的元组。

    返回:
    list of tuples: 每个元素是一个包含两个张量 (key, value) 的元组，
                    形状为 (batch_size, num_heads, sequence_length, head_dim)。
    """
    # 获取 batch_size 和层数
    batch_size = len(single_past_key_values_list)
    num_layers = len(single_past_key_values_list[0])
    
    # 初始化结果列表
    combined_past_key_values = []
    
    # 遍历每一层
    for layer_idx in range(num_layers):
        # 收集每个样本的 key 和 value
        keys = []
        values = []
        for sample_idx in range(batch_size):
            key, value = single_past_key_values_list[sample_idx][layer_idx]
            keys.append(key)
            values.append(value)
        
        # 将 keys 和 values 堆叠成 (batch_size, num_heads, sequence_length, head_dim) 的张量
        combined_key = torch.stack(keys, dim=0)
        combined_value = torch.stack(values, dim=0)
        
        # 添加到结果列表中
        combined_past_key_values.append((combined_key, combined_value))
    
    return combined_past_key_values