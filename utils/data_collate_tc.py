import init_utils

import torch,random

random.seed(42)

def get_aec_prompt():
    pre_prompt = """这是一个文本分类任务,7个类别中选择1个,类别标签如下:
0:direct类,所需的数据可以直接获得
1:indirect类,所需的数据需要额外推导和计算
2:method类,理解句子需要特定领域知识
3:reference类,理解句子需要参考其它条款
4:general类,为设计过程提供宏观指导
5:term类,定义规范中使用的术语
6:others类,通常是施工或维护要求,与设计无关,不属于上述6种类型
返回句子的类别标签:0,1,2,3,4,5,6中的一个,不要返回其他内容
要你进行分类的句子如下:\n"""
    post_prompt = """这个句子的类别是： """
    return pre_prompt,post_prompt

def get_tnews_prompt():
    pre_prompt = """这是一个文本分类任务,15个类别中选择1个,类别标签如下:
0:story
1:culture
2:entertainment
3:sports
4:finance
5:house
6:car
7:education
8:technology
9:military
a:travel
b:world
c:stock
d:agriculture
e:game
返回句子的类别标签:0,1,2,3,4,5,6,7,8,9,a,b,c,d,e中的一个,不要返回其他内容
要你进行分类的句子如下:\n"""
    post_prompt = """这个句子的类别是： """
    return pre_prompt,post_prompt

def get_agnews_prompt():
    pre_prompt = """This is a text classification task where one of four predefined categories must be selected. The category labels are as follows:
0:world
1:sports
2:business
3:science & technology
Output the category label of the sentence as one of: 0,1,2,or 3. Do not include any additional content.
 The sentence to be classified is as follows:\n"""
    post_prompt = """The category label is:"""
    return pre_prompt,post_prompt

def get_sst5_prompt():
    pre_prompt = """This is a text classification task where one of five predefined categories must be selected. The category labels are as follows:
0:very negative
1:negative
2:neutral
3:positive
4: very positive
Output the category label of the sentence as one of: 0,1,2,3,or 4. Do not include any additional content.
 The sentence to be classified is as follows:\n"""
    post_prompt = """The category label is: """
    return pre_prompt,post_prompt

def get_collate_fn_TC(data_format):
    collate_fns = {1: collate_fn_TC1,
                   201: collate_fn_TC201,
                   202: collate_fn_TC202,
                   203: collate_fn_TC203,
                   204: collate_fn_TC204,
                   3: collate_fn_TC3,
                   4: collate_fn_TC4,
                   5: collate_fn_TC5,
                   601: collate_fn_TC601,
                   602: collate_fn_TC602,
                   603: collate_fn_TC603,
                   604: collate_fn_TC604,
                   7: collate_fn_TC7,
                   8: collate_fn_TC8,
                    }
    return collate_fns[data_format]

def select_prompt_function(data_path):
    if "AEC_TC" in data_path:
        prompt_function = get_aec_prompt
    elif "TNEWS" in data_path:
        prompt_function = get_tnews_prompt
    elif "AG_NEWS" in data_path:
        prompt_function = get_agnews_prompt
    elif "SST5" in data_path:
        prompt_function = get_sst5_prompt

    return prompt_function
   
def collate_fn_TC1(data,tokenizer,max_length,split,data_path):
    """
    with prompt--no 
    seeing the answer--no
    discriminative or generative--discriminative
    """
    sents = [i['text'] for i in data]
    labels = [int(i['label']) if i['label'].isdigit() 
              else ord(i['label']) - ord('a') + 10  
              for i in data]

    inputs = tokenizer.batch_encode_plus(batch_text_or_text_pairs=sents,
                                   truncation=True,
                                   padding='longest',
                                   max_length=max_length,
                                   add_special_tokens=True,
                                   return_tensors='pt')
    labels = torch.LongTensor(labels)  
    # print(sents[0])
    return inputs,labels

def collate_fn_TC201(data,tokenizer,max_length,split,data_path):
    """
    with prompt--no 
    seeing the answer--yes
    discriminative or generative--discriminative
    """
    if split == "train":
        if ("AEC_TC" in data_path) or ("TNEWS" in data_path):
            sents = [i['text'] + "\n" + "类别是： " + i['label'] for i in data]
        elif ("AG_NEWS" in data_path) or ("SST5" in data_path):
            sents = [i['text'] + "\n" + "The category is: " + i['label'] for i in data]
    else:
        if ("AEC_TC" in data_path) or ("TNEWS" in data_path):
            sents = [i['text'] + "\n" + "类别是： " + tokenizer.mask_token for i in data]
        elif ("AG_NEWS" in data_path) or ("SST5" in data_path):
            sents = [i['text'] + "\n" + "The category is: " + tokenizer.mask_token for i in data]
    
    labels = [int(i['label']) if i['label'].isdigit() 
              else ord(i['label']) - ord('a') + 10  
              for i in data]

    inputs = tokenizer.batch_encode_plus(batch_text_or_text_pairs=sents,
                                   truncation=True,
                                   padding='longest',
                                   max_length=max_length,
                                   add_special_tokens=True,
                                   return_tensors='pt')
    labels = torch.LongTensor(labels)  
    # print(sents[0])
    return inputs,labels

def collate_fn_TC202(data,tokenizer,max_length,split,data_path):
    """
    with prompt--no 
    seeing the answer--yes
    discriminative or generative--discriminative
    """
    if split == "train":
        if "AEC_TC" in data_path:
            random_label = random.choice(["0","1","2","3","4","5","6"])
            sents = [i['text'] + "\n" + "类别是： " + random_label for i in data]
        elif "TNEWS" in data_path:
            random_label = random.choice(["0","1","2","3","4","5","6","7","8","9","a","b","c","d","e"])
            sents = [i['text'] + "\n" + "类别是： " + random_label for i in data]
        elif "AG_NEWS" in data_path:
            random_label = random.choice(["0","1","2","3"])
            sents = [i['text'] + "\n" + "The category is: " + random_label for i in data]
        elif "SST5" in data_path:
            random_label = random.choice(["0","1","2","3","4"])
            sents = [i['text'] + "\n" + "The category is: " + random_label for i in data]
    else:
        if ("AEC_TC" in data_path) or ("TNEWS" in data_path):
            sents = [i['text'] + "\n" + "类别是： " + tokenizer.mask_token for i in data]
        elif ("AG_NEWS" in data_path) or ("SST5" in data_path):
            sents = [i['text'] + "\n" + "The category is: " + tokenizer.mask_token for i in data]
    labels = [int(i['label']) if i['label'].isdigit() 
              else ord(i['label']) - ord('a') + 10  
              for i in data]

    inputs = tokenizer.batch_encode_plus(batch_text_or_text_pairs=sents,
                                   truncation=True,
                                   padding='longest',
                                   max_length=max_length,
                                   add_special_tokens=True,
                                   return_tensors='pt')
    labels = torch.LongTensor(labels)  
    # print(sents[0])
    return inputs,labels

def collate_fn_TC203(data,tokenizer,max_length,split,data_path):
    """
    with prompt--no 
    seeing the answer--yes
    discriminative or generative--discriminative
    """
    if split == "train":
        if ("AEC_TC" in data_path) or ("TNEWS" in data_path):
            sents = ["类别是： " + i['label'] + "\n" + i['text']  for i in data]
        elif ("AG_NEWS" in data_path) or ("SST5" in data_path):
            sents = ["The category is: " + i['label'] + "\n" + i['text']  for i in data]
    else:
        if ("AEC_TC" in data_path) or ("TNEWS" in data_path):
            sents = ["类别是： " + tokenizer.mask_token + "\n" + i['text']  for i in data]
        elif ("AG_NEWS" in data_path) or ("SST5" in data_path):
            sents = ["The category is: " + tokenizer.mask_token + "\n" + i['text']  for i in data]
    labels = [int(i['label']) if i['label'].isdigit() 
              else ord(i['label']) - ord('a') + 10  
              for i in data]

    inputs = tokenizer.batch_encode_plus(batch_text_or_text_pairs=sents,
                                   truncation=True,
                                   padding='longest',
                                   max_length=max_length,
                                   add_special_tokens=True,
                                   return_tensors='pt')
    labels = torch.LongTensor(labels)  
    # print(sents[0])
    return inputs,labels

def collate_fn_TC204(data,tokenizer,max_length,split,data_path):
    """
    with prompt--no 
    seeing the answer--yes
    discriminative or generative--discriminative
    """
    if split == "train":
        if "AEC_TC" in data_path:
            random_label = random.choice(["0","1","2","3","4","5","6"])
            sents = ["类别是： " + random_label + "\n" + i['text']  for i in data]
        elif "TNEWS" in data_path:
            random_label = random.choice(["0","1","2","3","4","5","6","7","8","9","a","b","c","d","e"])
            sents = ["类别是： " + random_label + "\n" + i['text']  for i in data]
        elif "AG_NEWS" in data_path:
            random_label = random.choice(["0","1","2","3"])
            sents = ["The category is: " + random_label + "\n" + i['text']  for i in data]
        elif "SST5" in data_path:
            random_label = random.choice(["0","1","2","3","4"])
            sents = ["The category is: " + random_label + "\n" + i['text']  for i in data]
    else:
        if ("AEC_TC" in data_path) or ("TNEWS" in data_path):
            sents = ["类别是： " + tokenizer.mask_token + "\n" + i['text']  for i in data]
        elif ("AG_NEWS" in data_path) or ("SST5" in data_path):
            sents = ["The category is: " + tokenizer.mask_token + "\n" + i['text']  for i in data]
    labels = [int(i['label']) if i['label'].isdigit() 
              else ord(i['label']) - ord('a') + 10  
              for i in data]

    inputs = tokenizer.batch_encode_plus(batch_text_or_text_pairs=sents,
                                   truncation=True,
                                   padding='longest',
                                   max_length=max_length,
                                   add_special_tokens=True,
                                   return_tensors='pt')
    labels = torch.LongTensor(labels)  
    # print(sents[0])
    return inputs,labels

def collate_fn_TC3(data,tokenizer,max_length,split,data_path):
    """
    with prompt--yes 
    seeing the answer--no
    discriminative or generative--discriminative
    """
    get_prompt = select_prompt_function(data_path)
    pre_prompt,post_prompt = get_prompt()

    sents = [pre_prompt + i['text']  + "\n" + post_prompt for i in data]
    labels = [int(i['label']) if i['label'].isdigit() 
              else ord(i['label']) - ord('a') + 10  
              for i in data]

    inputs = tokenizer.batch_encode_plus(batch_text_or_text_pairs=sents,
                                   truncation=True,
                                   padding='longest',
                                   max_length=max_length,
                                   add_special_tokens=True,
                                   return_tensors='pt')
    labels = torch.LongTensor(labels)  
    # print(sents[0])
    return inputs,labels

def collate_fn_TC4(data,tokenizer,max_length,split,data_path):
    """
    with prompt--yes 
    seeing the answer--yes
    discriminative or generative--discriminative
    """
    get_prompt = select_prompt_function(data_path)
    pre_prompt,post_prompt = get_prompt()
    if split == "train":
        sents = [pre_prompt + i['text']  + "\n" + post_prompt + i['label'] for i in data]
    else:
        sents = [pre_prompt + i['text']  + "\n" + post_prompt + tokenizer.mask_token for i in data]
    labels = [int(i['label']) if i['label'].isdigit() 
              else ord(i['label']) - ord('a') + 10  
              for i in data]

    inputs = tokenizer.batch_encode_plus(batch_text_or_text_pairs=sents,
                                   truncation=True,
                                   padding='longest',
                                   max_length=max_length,
                                   add_special_tokens=True,
                                   return_tensors='pt')
    labels = torch.LongTensor(labels)  
    # print(sents[0])
    return inputs,labels

def collate_fn_TC5(data,tokenizer,max_length,split,data_path):
    """
    with prompt--no 
    seeing the answer--no
    discriminative or generative--generative
    """
    if ("bert" in tokenizer.name_or_path) or ("gte-multilingual" in tokenizer.name_or_path):
        if ("AEC_TC" in data_path) or ("TNEWS" in data_path):
            sents = [i['text'] + "\n类别是： " + tokenizer.mask_token for i in data]
        elif ("AG_NEWS" in data_path) or ("SST5" in data_path):
            sents = [i['text'] + "\nThe category is: " + tokenizer.mask_token for i in data]
    elif ("Qwen" in tokenizer.name_or_path) or ("MiniCPM" in tokenizer.name_or_path):
        if ("AEC_TC" in data_path) or ("TNEWS" in data_path):
            sents = [i['text'] + "\n类别是： " + tokenizer.pad_token for i in data]
        elif ("AG_NEWS" in data_path) or ("SST5" in data_path):
            sents = [i['text'] + "\nThe category is: " + tokenizer.pad_token for i in data]
    
    if "MiniCPM" in tokenizer.name_or_path:
        label_for_target_token = [tokenizer.encode(i['label'],add_special_tokens=False)[-1] for i in data]
    else:
        label_for_target_token = [tokenizer.encode(i['label'],add_special_tokens=False)[0] for i in data]
    
    inputs = tokenizer.batch_encode_plus(batch_text_or_text_pairs=sents,
                                   truncation=True,
                                   padding='longest',
                                   max_length=max_length,
                                   add_special_tokens=True,
                                   return_tensors='pt')
    labels = torch.full(inputs["attention_mask"].shape,-100)
    for idx,label in enumerate(label_for_target_token):
        if ("bert" in tokenizer.name_or_path) or ("gte-multilingual" in tokenizer.name_or_path):
            mask_position = torch.where(inputs["input_ids"][idx] == tokenizer.mask_token_id)[0]
            labels[idx,mask_position] = label
        elif ("Qwen" in tokenizer.name_or_path) or ("MiniCPM" in tokenizer.name_or_path):
            last_one_position = torch.where(inputs["attention_mask"][idx] == 1)[0][-1]
            if last_one_position.item() < inputs["attention_mask"][idx].shape[0] - 1:
                inputs["input_ids"][idx,-1] = inputs["input_ids"][idx,last_one_position]
                inputs["input_ids"][idx,last_one_position] = tokenizer.pad_token_id
                inputs["attention_mask"][idx,last_one_position] = 0
                inputs["attention_mask"][idx,-1] = 1
            labels[idx,-1] = label
    labels = torch.LongTensor(labels)  
    # print(sents[0])
    return inputs,labels

def collate_fn_TC601(data,tokenizer,max_length,split,data_path):
    """
    with prompt--no 
    seeing the answer--yes
    discriminative or generative--generative
    """
    if ("bert" in tokenizer.name_or_path) or ("gte-multilingual" in tokenizer.name_or_path):
        if ("AEC_TC" in data_path) or ("TNEWS" in data_path):
            sents = [i['text'] + "\n类别是： " + tokenizer.mask_token for i in data]
        elif ("AG_NEWS" in data_path) or ("SST5" in data_path):
             sents = [i['text'] + "\nThe category is: " + tokenizer.mask_token for i in data]
    elif ("Qwen" in tokenizer.name_or_path) or ("MiniCPM" in tokenizer.name_or_path):
        if ("AEC_TC" in data_path) or ("TNEWS" in data_path):
            sents = [i['text'] + "\n类别是： " + tokenizer.pad_token for i in data]
        elif ("AG_NEWS" in data_path) or ("SST5" in data_path):
            sents = [i['text'] + "\nThe category is: " + tokenizer.pad_token for i in data]

    if "MiniCPM" in tokenizer.name_or_path:
        label_for_target_token = [tokenizer.encode(i['label'],add_special_tokens=False)[-1] for i in data]
    else:
        label_for_target_token = [tokenizer.encode(i['label'],add_special_tokens=False)[0] for i in data]
    
    inputs = tokenizer.batch_encode_plus(batch_text_or_text_pairs=sents,
                                   truncation=True,
                                   padding='longest',
                                   max_length=max_length,
                                   add_special_tokens=True,
                                   return_tensors='pt')
    labels = torch.full(inputs["attention_mask"].shape,-100)

    for idx,label in enumerate(label_for_target_token):
        input_ids = inputs["input_ids"][idx]
        if ("bert" in tokenizer.name_or_path) or ("gte-multilingual" in tokenizer.name_or_path): 
            first_mask_position = (input_ids == tokenizer.mask_token_id).nonzero(as_tuple=True)[0][0]
            if split == "train":
                input_ids[first_mask_position] = label
            labels[idx,first_mask_position] = label
        elif "Qwen" in tokenizer.name_or_path:
            first_pad_position = (input_ids == tokenizer.pad_token_id).nonzero(as_tuple=True)[0][0]
            if split == "train":
                input_ids[first_pad_position] = label
            last_one_position = torch.where(inputs["attention_mask"][idx] == 1)[0][-1]
            if last_one_position.item() < inputs["attention_mask"][idx].shape[0] - 1:
                inputs["input_ids"][idx,-1] = inputs["input_ids"][idx,last_one_position]
                inputs["input_ids"][idx,last_one_position] = tokenizer.pad_token_id
                inputs["attention_mask"][idx,last_one_position] = 0
                inputs["attention_mask"][idx,-1] = 1
            labels[idx,-1] = label
        elif "MiniCPM" in tokenizer.name_or_path: 
            if split == "train":
                input_ids[-1] = label
            labels[idx,-1] = label
        
    labels = torch.LongTensor(labels)  
    # print(tokenizer.decode(inputs["input_ids"][0]))
    return inputs,labels

def collate_fn_TC602(data,tokenizer,max_length,split,data_path):
    """
    with prompt--no 
    seeing the answer--yes
    discriminative or generative--generative
    """
    # 获取random_label
    if "AEC_TC" in data_path:
        random_label = random.choice(["0","1","2","3","4","5","6"])
    elif "TNEWS" in data_path:
        random_label = random.choice(["0","1","2","3","4","5","6","7","8","9","a","b","c","d","e"])
    elif "AG_NEWS" in data_path:
        random_label = random.choice(["0","1","2","3"])
    elif "SST5" in data_path:
        random_label = random.choice(["0","1","2","3","4"])
    if "MiniCPM" in tokenizer.name_or_path:
        random_label = tokenizer.encode(random_label,add_special_tokens=False)[-1]
    else:
        random_label = tokenizer.encode(random_label,add_special_tokens=False)[0]

    if ("bert" in tokenizer.name_or_path) or ("gte-multilingual" in tokenizer.name_or_path):
        if ("AEC_TC" in data_path) or ("TNEWS" in data_path):
            sents = [i['text'] + "\n类别是： " + tokenizer.mask_token for i in data]
        elif ("AG_NEWS" in data_path) or ("SST5" in data_path):
             sents = [i['text'] + "\nThe category is: " + tokenizer.mask_token for i in data]
    elif ("Qwen" in tokenizer.name_or_path) or ("MiniCPM" in tokenizer.name_or_path):
        if ("AEC_TC" in data_path) or ("TNEWS" in data_path):
            sents = [i['text'] + "\n类别是： " + tokenizer.pad_token for i in data]
        elif ("AG_NEWS" in data_path) or ("SST5" in data_path):
            sents = [i['text'] + "\nThe category is: " + tokenizer.pad_token for i in data]

    if "MiniCPM" in tokenizer.name_or_path:
        label_for_target_token = [tokenizer.encode(i['label'],add_special_tokens=False)[-1] for i in data]
    else:
        label_for_target_token = [tokenizer.encode(i['label'],add_special_tokens=False)[0] for i in data]
    
    inputs = tokenizer.batch_encode_plus(batch_text_or_text_pairs=sents,
                                   truncation=True,
                                   padding='longest',
                                   max_length=max_length,
                                   add_special_tokens=True,
                                   return_tensors='pt')
    labels = torch.full(inputs["attention_mask"].shape,-100)

    for idx,label in enumerate(label_for_target_token):
        input_ids = inputs["input_ids"][idx]
        if ("bert" in tokenizer.name_or_path) or ("gte-multilingual" in tokenizer.name_or_path): 
            first_mask_position = (input_ids == tokenizer.mask_token_id).nonzero(as_tuple=True)[0][0]
            if split == "train":
                input_ids[first_mask_position] = random_label
            labels[idx,first_mask_position] = label
        elif "Qwen" in tokenizer.name_or_path:
            first_pad_position = (input_ids == tokenizer.pad_token_id).nonzero(as_tuple=True)[0][0]
            if split == "train":
                input_ids[first_pad_position] = random_label
            last_one_position = torch.where(inputs["attention_mask"][idx] == 1)[0][-1]
            if last_one_position.item() < inputs["attention_mask"][idx].shape[0] - 1:
                inputs["input_ids"][idx,-1] = inputs["input_ids"][idx,last_one_position]
                inputs["input_ids"][idx,last_one_position] = tokenizer.pad_token_id
                inputs["attention_mask"][idx,last_one_position] = 0
                inputs["attention_mask"][idx,-1] = 1
            labels[idx,-1] = label
        elif "MiniCPM" in tokenizer.name_or_path:
            if split == "train":
                input_ids[-1] = random_label
            labels[idx,-1] = label
        
    labels = torch.LongTensor(labels)  
    # print(tokenizer.decode(inputs["input_ids"][0]))
    return inputs,labels

def collate_fn_TC603(data,tokenizer,max_length,split,data_path):
    """
    with prompt--no 
    seeing the answer--yes
    discriminative or generative--generative
    """
    if ("bert" in tokenizer.name_or_path) or ("gte-multilingual" in tokenizer.name_or_path):
        if ("AEC_TC" in data_path) or ("TNEWS" in data_path):
            sents = ["已知类别是： " + tokenizer.mask_token + "\n" + i['text'] + "\n预测类别是： " + tokenizer.mask_token for i in data]
        elif ("AG_NEWS" in data_path) or ("SST5" in data_path):
            sents = ["The known category is: " + tokenizer.mask_token + "\n" + i['text'] + "\nThe predicted category is: " + tokenizer.mask_token for i in data]
    elif ("Qwen" in tokenizer.name_or_path) or ("MiniCPM" in tokenizer.name_or_path):
        if ("AEC_TC" in data_path) or ("TNEWS" in data_path):
            sents = ["已知类别是： " + tokenizer.pad_token + "\n" + i['text'] + "\n预测类别是： " + tokenizer.pad_token for i in data]
        elif ("AG_NEWS" in data_path) or ("SST5" in data_path):
            sents = ["The known category is: " + tokenizer.pad_token + "\n" + i['text'] + "\nThe predicted category is: " + tokenizer.pad_token for i in data]

    if "MiniCPM" in tokenizer.name_or_path:
        label_for_target_token = [tokenizer.encode(i['label'],add_special_tokens=False)[-1] for i in data]
    else:
        label_for_target_token = [tokenizer.encode(i['label'],add_special_tokens=False)[0] for i in data]
    
    inputs = tokenizer.batch_encode_plus(batch_text_or_text_pairs=sents,
                                   truncation=True,
                                   padding='longest',
                                   max_length=max_length,
                                   add_special_tokens=True,
                                   return_tensors='pt')
    labels = torch.full(inputs["attention_mask"].shape,-100)

    for idx,label in enumerate(label_for_target_token):
        input_ids = inputs["input_ids"][idx]
        if ("bert" in tokenizer.name_or_path) or ("gte-multilingual" in tokenizer.name_or_path): 
            first_mask_position,second_mask_position = (input_ids == tokenizer.mask_token_id).nonzero(as_tuple=True)[0]
            if split == "train":
                input_ids[first_mask_position] = label
            labels[idx,second_mask_position] = label
        elif "Qwen" in tokenizer.name_or_path:
            first_pad_position = (input_ids == tokenizer.pad_token_id).nonzero(as_tuple=True)[0][0]
            if split == "train":
                input_ids[first_pad_position] = label            
            
            last_one_position = torch.where(inputs["attention_mask"][idx] == 1)[0][-1]
            if last_one_position.item() < inputs["attention_mask"][idx].shape[0] - 1:
                inputs["input_ids"][idx,-1] = inputs["input_ids"][idx,last_one_position]
                inputs["input_ids"][idx,last_one_position] = tokenizer.pad_token_id
                inputs["attention_mask"][idx,last_one_position] = 0
                inputs["attention_mask"][idx,-1] = 1
            labels[idx,-1] = label
    
    labels = torch.LongTensor(labels)  
    # print(tokenizer.decode(inputs["input_ids"][0]))
    return inputs,labels

def collate_fn_TC604(data,tokenizer,max_length,split,data_path):
    """
    with prompt--no 
    seeing the answer--yes
    discriminative or generative--generative
    """
    # 获取random_label
    if "AEC_TC" in data_path:
        random_label = random.choice(["0","1","2","3","4","5","6"])
    elif "TNEWS" in data_path:
        random_label = random.choice(["0","1","2","3","4","5","6","7","8","9","a","b","c","d","e"])
    elif "AG_NEWS" in data_path:
        random_label = random.choice(["0","1","2","3"])
    elif "SST5" in data_path:
        random_label = random.choice(["0","1","2","3","4"])
    if "MiniCPM" in tokenizer.name_or_path:
        random_label = tokenizer.encode(random_label,add_special_tokens=False)[-1]
    else:
        random_label = tokenizer.encode(random_label,add_special_tokens=False)[0]

    if ("bert" in tokenizer.name_or_path) or ("gte-multilingual" in tokenizer.name_or_path):
        if ("AEC_TC" in data_path) or ("TNEWS" in data_path):
            sents = ["已知类别是： " + tokenizer.mask_token + "\n" + i['text'] + "\n预测类别是： " + tokenizer.mask_token for i in data]
        elif ("AG_NEWS" in data_path) or ("SST5" in data_path):
            sents = ["The known category is: " + tokenizer.mask_token + "\n" + i['text'] + "\nThe predicted category is: " + tokenizer.mask_token for i in data]
    elif ("Qwen" in tokenizer.name_or_path) or ("MiniCPM" in tokenizer.name_or_path):
        if ("AEC_TC" in data_path) or ("TNEWS" in data_path):
            sents = ["已知类别是： " + tokenizer.pad_token + "\n" + i['text'] + "\n预测类别是： " + tokenizer.pad_token for i in data]
        elif ("AG_NEWS" in data_path) or ("SST5" in data_path):
            sents = ["The known category is: " + tokenizer.pad_token + "\n" + i['text'] + "\nThe predicted category is: " + tokenizer.pad_token for i in data]

    if "MiniCPM" in tokenizer.name_or_path:
        label_for_target_token = [tokenizer.encode(i['label'],add_special_tokens=False)[-1] for i in data]
    else:
        label_for_target_token = [tokenizer.encode(i['label'],add_special_tokens=False)[0] for i in data]
    
    inputs = tokenizer.batch_encode_plus(batch_text_or_text_pairs=sents,
                                   truncation=True,
                                   padding='longest',
                                   max_length=max_length,
                                   add_special_tokens=True,
                                   return_tensors='pt')
    labels = torch.full(inputs["attention_mask"].shape,-100)

    for idx,label in enumerate(label_for_target_token):
        input_ids = inputs["input_ids"][idx]
        if ("bert" in tokenizer.name_or_path) or ("gte-multilingual" in tokenizer.name_or_path): 
            first_mask_position,second_mask_position = (input_ids == tokenizer.mask_token_id).nonzero(as_tuple=True)[0]
            if split == "train":
                input_ids[first_mask_position] = random_label
            labels[idx,second_mask_position] = label
        elif "Qwen" in tokenizer.name_or_path:
            first_pad_position = (input_ids == tokenizer.pad_token_id).nonzero(as_tuple=True)[0][0]
            if split == "train":
                input_ids[first_pad_position] = random_label
            last_one_position = torch.where(inputs["attention_mask"][idx] == 1)[0][-1]
            if last_one_position.item() < inputs["attention_mask"][idx].shape[0] - 1:
                inputs["input_ids"][idx,-1] = inputs["input_ids"][idx,last_one_position]
                inputs["input_ids"][idx,last_one_position] = tokenizer.pad_token_id
                inputs["attention_mask"][idx,last_one_position] = 0
                inputs["attention_mask"][idx,-1] = 1
            labels[idx,-1] = label
        
    labels = torch.LongTensor(labels)  
    # print(tokenizer.decode(inputs["input_ids"][0]))
    return inputs,labels

def collate_fn_TC7(data,tokenizer,max_length,split,data_path):
    """
    with prompt--no 
    seeing the answer--no
    discriminative or generative--generative
    """
    get_prompt = select_prompt_function(data_path)
    pre_prompt,post_prompt = get_prompt()
    
    if ("bert" in tokenizer.name_or_path) or ("gte-multilingual" in tokenizer.name_or_path):
        sents = [pre_prompt + i['text'] + "\n" + post_prompt + tokenizer.mask_token for i in data]
    elif ("Qwen" in tokenizer.name_or_path) or ("MiniCPM" in tokenizer.name_or_path):
        sents = [pre_prompt + i['text'] + "\n" + post_prompt + tokenizer.pad_token for i in data]
    
    if "MiniCPM" in tokenizer.name_or_path:
        label_for_target_token = [tokenizer.encode(i['label'],add_special_tokens=False)[-1] for i in data]
    else:
        label_for_target_token = [tokenizer.encode(i['label'],add_special_tokens=False)[0] for i in data]
    
    inputs = tokenizer.batch_encode_plus(batch_text_or_text_pairs=sents,
                                   truncation=True,
                                   padding='longest',
                                   max_length=max_length,
                                   add_special_tokens=True,
                                   return_tensors='pt')
    labels = torch.full(inputs["attention_mask"].shape,-100)
    
    for idx,label in enumerate(label_for_target_token):
        if ("bert" in tokenizer.name_or_path) or ("gte-multilingual" in tokenizer.name_or_path):
            mask_position = torch.where(inputs["input_ids"][idx] == tokenizer.mask_token_id)[0]
            labels[idx,mask_position] = label
        elif "Qwen" in tokenizer.name_or_path:
            last_one_position = torch.where(inputs["attention_mask"][idx] == 1)[0][-1]
            if last_one_position.item() < inputs["attention_mask"][idx].shape[0] - 1:
                inputs["input_ids"][idx,-1] = inputs["input_ids"][idx,last_one_position]
                inputs["input_ids"][idx,last_one_position] = tokenizer.pad_token_id
                inputs["attention_mask"][idx,last_one_position] = 0
                inputs["attention_mask"][idx,-1] = 1
            labels[idx,-1] = label
        elif "MiniCPM" in tokenizer.name_or_path:
            if split == "train":
                inputs["input_ids"][-1] = label
            labels[idx,-1] = label
    labels = torch.LongTensor(labels)  
    # print(tokenizer.decode(inputs["input_ids"][0]))    
    return inputs,labels

def collate_fn_TC8(data,tokenizer,max_length,split,data_path):
    """
    with prompt--no 
    seeing the answer--no
    discriminative or generative--generative
    """
    get_prompt = select_prompt_function(data_path)
    pre_prompt,post_prompt = get_prompt()
    
    if ("bert" in tokenizer.name_or_path) or ("gte-multilingual" in tokenizer.name_or_path):
        if ("AEC_TC" in data_path) or ("TNEWS" in data_path):
            sents = [pre_prompt + i['text'] + "\n" + post_prompt + tokenizer.mask_token  for i in data]
        elif ("AG_NEWS" in data_path) or ("SST5" in data_path):
            sents = [pre_prompt + i['text'] + "\n" + post_prompt + tokenizer.mask_token  for i in data]            
    elif ("Qwen" in tokenizer.name_or_path) or ("MiniCPM" in tokenizer.name_or_path):
        sents = [pre_prompt + i['text'] + "\n" + post_prompt + tokenizer.pad_token for i in data]
    
    if "MiniCPM" in tokenizer.name_or_path:
        label_for_target_token = [tokenizer.encode(i['label'],add_special_tokens=False)[-1] for i in data]
    else:
        label_for_target_token = [tokenizer.encode(i['label'],add_special_tokens=False)[0] for i in data]
    
    inputs = tokenizer.batch_encode_plus(batch_text_or_text_pairs=sents,
                                   truncation=True,
                                   padding='longest',
                                   max_length=max_length,
                                   add_special_tokens=True,
                                   return_tensors='pt')
    labels = torch.full(inputs["attention_mask"].shape,-100)

    for idx,label in enumerate(label_for_target_token):
        input_ids = inputs["input_ids"][idx]
        if ("bert" in tokenizer.name_or_path) or ("gte-multilingual" in tokenizer.name_or_path): 
            first_mask_position = (input_ids == tokenizer.mask_token_id).nonzero(as_tuple=True)[0][0]
            if split == "train":
                input_ids[first_mask_position] = label
            labels[idx,first_mask_position] = label
        
        elif "Qwen" in tokenizer.name_or_path:
            first_pad_position = (input_ids == tokenizer.pad_token_id).nonzero(as_tuple=True)[0][0]
            if split == "train":
                input_ids[first_pad_position] = label
            last_one_position = torch.where(inputs["attention_mask"][idx] == 1)[0][-1]
            if last_one_position.item() < inputs["attention_mask"][idx].shape[0] - 1:
                inputs["input_ids"][idx,-1] = inputs["input_ids"][idx,last_one_position]
                inputs["input_ids"][idx,last_one_position] = tokenizer.pad_token_id
                inputs["attention_mask"][idx,last_one_position] = 0
                inputs["attention_mask"][idx,-1] = 1
            labels[idx,-1] = label
        
        elif "MiniCPM" in tokenizer.name_or_path:
            if split == "train":
                input_ids[-1] = label
            labels[idx,-1] = label
    # print(tokenizer.decode(inputs["input_ids"][0]))     
    labels = torch.LongTensor(labels)  
    
    return inputs,labels
