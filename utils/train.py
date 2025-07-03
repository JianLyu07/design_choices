import init_utils,os,sys

import math,logging,torch
from torch.optim import AdamW
from tqdm import tqdm

from model_proc import load_model,prepare_model,test_TC,compute_metrics_mutiple
from data_proc import get_dataloaders,sanity_check
from accelerate import Accelerator

from basic_function import get_abs_path

def train(model,data_loader,optimizer,accelerator,epoch_name,g_args):
    clr_epoch = {} 
    loss_train_epoch = {} 
    result_train_epoch = {} 
    loss_test_epoch = {} 
    result_test_epoch = {} 
    if "val" in data_loader.keys():    
        loss_val_epoch = {} 
        result_val_epoch = {} 
    stage_break_flag = False 
    rbl_break_flag = False 
    
    loss_train_checkstep = [] 
    predictions_checkstep = []
    labels_checkstep = [] 

    test_model = test_TC

    for i, (inputs, labels) in tqdm(enumerate(data_loader["train"])):
        output = model(labels = labels,**inputs)        
        loss = output["loss"]        
        accelerator.backward(loss)
        optimizer.step()
        optimizer.zero_grad() 
         
        loss_train_checkstep.append(loss.item())

        # get prediction and labels
        if g_args["data_format"] in [1,201,202,203,204,3,4]:
            logits = output['logits']
        elif g_args["data_format"] in [5,601,602,603,604,7,8]:
            mask = labels != -100
            relevant_indices = labels.argmax(dim=1)
            if ("Qwen" in g_args["model_path"]) or ("MiniCPM" in g_args["model_path"]): 
                relevant_indices = relevant_indices - 1 
            logits = output["logits"].gather(dim=1, 
                                    index=relevant_indices.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, output["logits"].size(-1)))
            logits = logits.squeeze(1)
            labels = labels[mask].view(-1)   
        
        predictions_checkstep = predictions_checkstep + logits.argmax(dim=1).tolist()
        labels_checkstep = labels_checkstep + labels.tolist()
        
        if (i+1) % int(len(data_loader["train"])/g_args["num_checks"]) == 0:
            # current cs--check_step
            current_cs = (i+1) // int(len(data_loader["train"])/g_args["num_checks"])
                        
            # print current_checkstep
            checkstep_name = epoch_name + "_checkstep_" + str(current_cs)
            logging.info(checkstep_name)

            # Record and print the learning rate for the current check_step
            clr_epoch[checkstep_name] = optimizer.state_dict()['param_groups'][0]['lr']
            logging.info(f"current_learning_rate={clr_epoch[checkstep_name]}")
            
            # train resut
            loss_train_epoch[checkstep_name] = (sum(loss_train_checkstep) / (len(loss_train_checkstep)+1e-6)) #current checkstep的loss_train平均值
            result_train_epoch[checkstep_name] = compute_metrics_mutiple(predictions_checkstep,labels_checkstep)
            loss_train_checkstep = [] 
            predictions_checkstep = []
            labels_checkstep = [] 
            logging.info(f"train--{g_args['target_metric']}={result_train_epoch[checkstep_name][g_args['target_metric']]:.4f}")
            logging.info(f"train--accuracy={result_train_epoch[checkstep_name]['accuracy']:.4f}")
            logging.info(f"train--train_loss={loss_train_epoch[checkstep_name]:.4f}")
            
            # test resut    
            eval_test_checkstep = test_model(model,data_loader["test"],g_args)
            loss_test_epoch[checkstep_name] = eval_test_checkstep["loss"]
            result_test_epoch[checkstep_name] = compute_metrics_mutiple(eval_test_checkstep['predictions'],eval_test_checkstep['labels'])
            logging.info(f"test--{g_args['target_metric']}={result_test_epoch[checkstep_name][g_args['target_metric']]:.4f}")
            logging.info(f"test--accuracy={result_test_epoch[checkstep_name]['accuracy']:.4f}")
            logging.info(f"test--test_loss={loss_test_epoch[checkstep_name]:.4f}")
            
            # validation result
            if "val" in data_loader.keys(): 
                eval_val_checkstep = test_model(model,data_loader["val"],g_args)
                loss_val_epoch[checkstep_name] = eval_val_checkstep["loss"]
                result_val_epoch[checkstep_name] = compute_metrics_mutiple(eval_val_checkstep['predictions'],eval_val_checkstep['labels'])
                logging.info(f"val--{g_args['target_metric']}={result_val_epoch[checkstep_name][g_args['target_metric']]:.4f}")
                logging.info(f"val--accuracy={result_val_epoch[checkstep_name]['accuracy']:.4f}")
                logging.info(f"val--val_loss={loss_val_epoch[checkstep_name]:.4f}")
                                                
            # If the loss on the test or validation set becomes NaN or Inf, stop training for this configuration immediately
            if math.isnan(loss_test_epoch[checkstep_name]) or math.isinf(loss_test_epoch[checkstep_name]):
                rbl_break_flag = True
                break
            if "val" in data_loader.keys(): 
                if math.isnan(loss_val_epoch[checkstep_name]) or math.isinf(loss_val_epoch[checkstep_name]):
                    rbl_break_flag = True
                    break
            
            # Determine whether to save the model: save the model if either the validation or test set metric exceeds target_save_1, or if both the validation and test set metrics exceed target_save_2
            save_flag = False
            if result_test_epoch[checkstep_name][g_args['target_metric']] > g_args["target_save_1"]:
                save_flag = True
                save_name = checkstep_name + f"_t_{g_args['target_metric']}_{result_test_epoch[checkstep_name][g_args['target_metric']]:.4f}"
            if "val" in data_loader.keys(): 
                if result_val_epoch[checkstep_name][g_args['target_metric']] > g_args["target_save_1"]:
                    save_flag = True 
                elif (result_test_epoch[checkstep_name][g_args['target_metric']] > g_args["target_save_2"] and 
                      result_val_epoch[checkstep_name][g_args['target_metric']] > g_args["target_save_2"]):
                    save_flag = True
                save_name = checkstep_name + f"_t_{g_args['target_metric']}_{result_test_epoch[checkstep_name][g_args['target_metric']]:.4f}"
                save_name = save_name + f"_v_{g_args['target_metric']}_{result_val_epoch[checkstep_name][g_args['target_metric']]:.4f}"
            if save_flag == True:
                model.save_pretrained(get_abs_path("../../model_saved/"+f'{save_name}'))
                
                model.train()
                
    result_epoch = {
        "clr_epoch": clr_epoch,
        "loss_train_epoch": loss_train_epoch,
        "result_train_epoch": result_train_epoch,
        "loss_test_epoch": loss_test_epoch,
        "result_test_epoch": result_test_epoch,
        "stage_break_flag": stage_break_flag,
        "rbl_break_flag": rbl_break_flag
    }
    if "val" in data_loader.keys():
        result_epoch.update({
            "loss_val_epoch": loss_val_epoch,
            "result_val_epoch": result_val_epoch
        })
    return result_epoch

def pipeline_tuning(g_args,lora_r,batchsize,lr):
    accelerator = Accelerator()
    if g_args["finetune_pattern"] == "lora":
        setting = f"lora_r = {lora_r}__batchsize = {batchsize}__starting_learning_rate = {lr}"
    elif g_args["finetune_pattern"] == "full":
        setting = f"full_finetune___batchsize = {batchsize}__starting_learning_rate = {lr}"
    logging.info(f"*******当前设置{setting}*******")
       
    # get data_loader
    data_loader = get_dataloaders(g_args=g_args,batchsize=batchsize)
    data_loader["train"] =  accelerator.prepare(data_loader["train"])    
    data_loader["test"] = accelerator.prepare(data_loader["test"])
    if "val" in data_loader.keys():
        data_loader["val"] = accelerator.prepare(data_loader["val"])
    sanity_check(data_loader["train"],g_args["tokenizer"],0,0)
    sanity_check(data_loader["test"],g_args["tokenizer"],0,0)
    
    # load model
    model = load_model(g_args = g_args)
    model = accelerator.prepare(model)

    clr_setting = {} 
    loss_train_setting = {}
    result_train_setting = {}
    loss_test_setting = {} 
    result_test_setting = {} 
    if "val" in data_loader.keys():    
        loss_val_setting = {} 
        result_val_setting = {} 
    
    optimizer1 = AdamW(model.parameters(), lr=lr, weight_decay=g_args["weight_decay"])
    optimizer1 = accelerator.prepare(optimizer1)
    if g_args["lr_1"] is not None:
        optimizer2 = AdamW(model.parameters(), lr=g_args["lr_1"], weight_decay=g_args["weight_decay"])
        optimizer2 = accelerator.prepare(optimizer2)
    
    for i,tuning_stage in enumerate(g_args["tuning_stages"]):

        model = prepare_model(model,tuning_stage,g_args)

        if tuning_stage == 1 and g_args["lr_1"] is not None:
            optimizer = optimizer2
        else:
            optimizer = optimizer1

        for epoch in range(g_args["epochs"][i]):           
            if g_args["finetune_pattern"] == "lora":
                epoch_name = f"r_{lora_r}_bs_{batchsize}_slr_{lr}_ts_{tuning_stage}_ep_{epoch}" 
            elif g_args["finetune_pattern"] == "full":
                epoch_name = f"full_finetune_bs_{batchsize}_slr_{lr}_ts_{tuning_stage}_ep_{epoch}"

            result_epoch = train(model=model,
                                data_loader=data_loader,
                                optimizer=optimizer,
                                accelerator=accelerator,
                                epoch_name = epoch_name,
                                g_args = g_args)
 
            clr_setting = {**clr_setting, **result_epoch["clr_epoch"]}
            loss_train_setting = {**loss_train_setting, **result_epoch["loss_train_epoch"]} 
            result_train_setting = {**result_train_setting, **result_epoch["result_train_epoch"]} 
            loss_test_setting = {**loss_test_setting, **result_epoch["loss_test_epoch"]} 
            result_test_setting = {**result_test_setting, **result_epoch["result_test_epoch"]} 
            if "val" in data_loader.keys():    
                loss_val_setting = {**loss_val_setting, **result_epoch["loss_val_epoch"]} 
                result_val_setting = {**result_val_setting, **result_epoch["result_val_epoch"]}
            
            stage_break_flag = result_epoch["stage_break_flag"]
            if stage_break_flag == True:
                break
            
            rbl_break_flag = result_epoch["rbl_break_flag"]
            if rbl_break_flag == True:
                break
              
        if rbl_break_flag == True:
                break     
    
    result_setting = {
        "clr_setting": clr_setting,
        "loss_train_setting": loss_train_setting,
        "result_train_setting": result_train_setting,
        "loss_test_setting": loss_test_setting,
        "result_test_setting": result_test_setting
    }
    if "val" in data_loader.keys():
        result_setting.update({
            "loss_val_setting": loss_val_setting,
            "result_val_setting": result_val_setting
        })  

    del model,data_loader,optimizer1,accelerator
    torch.cuda.empty_cache()
    
    return result_setting