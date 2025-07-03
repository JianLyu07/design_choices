import init_utils

import logging
from torch.utils.data import DataLoader
from datasets import load_from_disk

from data_collate_tc import get_collate_fn_TC
from basic_function import get_abs_path
    
def reshape_and_remove_pad(out, labels):
    """
    Reshape the computed results and labels, and remove the padding
    """
    out = out.reshape(-1, out.shape[-1])
    labels = labels.reshape(-1).detach()
    select = labels != -100 
    out = out[select]
    labels = labels[select]
    return out, labels

def get_dataloaders(g_args,batchsize):
    '''
    Get the dataloader
    Obtain the dataset path and data processing function through the input parameter g_args
    Get the batch size for loading through the input parameter batchsize
    '''
    dataset = load_from_disk(get_abs_path(g_args["data_path"])).shuffle(seed=42)
    # Use only a subset of the data for process testing
    if g_args["is_attempt"] == True:
        dataset["train"] = dataset["train"].select(range(64))
        dataset["test"] = dataset["test"].select(range(32))
        if "validation" in dataset.keys():
            dataset["validation"] = dataset["validation"].select(range(32))

    collate_fn = get_collate_fn_TC(g_args["data_format"])

    train_loader = DataLoader(dataset = dataset["train"],
                            batch_size = batchsize,
                            collate_fn=lambda data : collate_fn(data = data,
                                                        tokenizer = g_args["tokenizer"],
                                                        max_length = g_args["max_length"],
                                                        split = "train",
                                                        data_path = g_args["data_path"]), 
                            shuffle=True,
                            drop_last=False)
    test_loader = DataLoader(dataset = dataset["test"],
                            batch_size = batchsize,
                            collate_fn=lambda data : collate_fn(data = data,
                                                        tokenizer = g_args["tokenizer"],
                                                        max_length = g_args["max_length"],
                                                        split = "test",
                                                        data_path = g_args["data_path"]),
                            shuffle=False,
                            drop_last=False)
    dataloaders = {"train":train_loader,"test":test_loader}
    if "validation" in dataset.keys():
        val_loader = DataLoader(dataset = dataset["validation"],
                            batch_size = batchsize,
                            collate_fn=lambda data : collate_fn(data = data,
                                                        tokenizer = g_args["tokenizer"],
                                                        max_length = g_args["max_length"],
                                                        split = "validation",
                                                        data_path = g_args["data_path"]),
                            shuffle=False,
                            drop_last=False)
        dataloaders["val"] = val_loader

    return dataloaders    

def sanity_check(data_loader,tokenizer,m,n):
    """
    检查某个torch.utils.data.DataLoader的第m批次，第n条数据
    例如：sanity_check(train_loader,0,0)
    """
    for i,(inputs,labels) in enumerate(data_loader):  
        if i == m:
            logging.info(40*"*"+"Sanity Check"+40*"*")
            title = "decoded input_id:input_id-->labels"
            for key in inputs.keys():
                if key != "input_ids":
                    title = title + f"-->{key}"
            logging.info(title)
            for j in range(len(inputs['input_ids'][n])):
                message = ""
                if labels.ndim == 2:
                    message = f'{tokenizer.decode(inputs["input_ids"][n][j])}:{inputs["input_ids"][n][j].item()}-->{labels[n][j].item()}'
                elif labels.ndim == 1:
                    message = f'{tokenizer.decode(inputs["input_ids"][n][j])}:{inputs["input_ids"][n][j].item()}-->{labels[n].item()}'
                for key in inputs.keys():
                    if key != "input_ids":
                        message = message + f"-->{inputs[key][n][j]}"
                logging.info(message)
            break
    logging.info(40*"*"+"Sanity Check"+40*"*")   