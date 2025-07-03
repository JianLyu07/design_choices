import init_utils

import torch,logging
import numpy as np
from transformers import AutoModelForSequenceClassification,AutoModelForMaskedLM,AutoModelForCausalLM
from peft import get_peft_model,LoraConfig
from torch import nn
from sklearn.metrics import accuracy_score, f1_score

from basic_function import get_abs_path

class LayerToFloat32(nn.Module):
    def __init__(self, model_layer):
        super(LayerToFloat32, self).__init__()
        # All components that require training need to use float32      
        self.model_layer = model_layer.to(torch.float32)
        # Set to require gradient backpropagation
        for param in self.model_layer.parameters():
            param.requires_grad = True
 
    def forward(self, x):
        x=x.to(torch.float32).cuda()
        y=self.model_layer(x)
        return y.half()
  
def layer_params_init(model_layer):
    """
    Initialize the parameters of untrained layers and set them to trainable
    """
    for param in model_layer.parameters():
        nn.init.normal_(param, mean=0.0, std=0.01)
        param.requires_grad = True
    return model_layer           

def check_parameters(model):
    """
    Print the number of trainable and non-trainable parameters in the model
    Check that the data type of trainable parameters is float32, and the data type of non-trainable parameters is float16
    """
    trainable_parameters = 0
    untrainable_parameters = 0
    for name,param in model.named_parameters():
        if param.requires_grad == True:
            trainable_parameters = trainable_parameters + param.nelement()
            if str(param.dtype) != "torch.float32" and str(param.dtype) != "torch.bfloat32":
                logging.info(f"{name} --should check the dtype-- {param.requires_grad} {param.dtype}")
        else:
            untrainable_parameters = untrainable_parameters + param.nelement()
            if str(param.dtype) != "torch.float16" and str(param.dtype) != "torch.bfloat16":
                logging.info(f"{name} --should check the dtype-- {param.requires_grad} {param.dtype}")
    logging.info("Trainable："+str(trainable_parameters)) 
    logging.info("Untrainable："+str(untrainable_parameters))
    logging.info("Total："+str(trainable_parameters+untrainable_parameters))
    
def test_TC(model,data_loader,g_args):
    """
    Retrieve the predictions and labels of the model on the dataset
    The data_loader is an instance of the torch.utils.data.DataLoader class
    """    
    model.eval()
    predictions_sum = []
    labels_sum = []
    loss = []
    with torch.no_grad():
        for i, (inputs,labels) in enumerate(data_loader):
            output = model(labels=labels,**inputs)
            
            if output['logits'].ndim == 2:
                logits = output['logits']
            elif output['logits'].ndim == 3:
                mask = labels != -100
                relevant_indices = labels.argmax(dim=1)
                if ("Qwen" in g_args["model_path"]) or ("MiniCPM" in g_args["model_path"]): 
                    relevant_indices = relevant_indices - 1 
                logits = output["logits"].gather(dim=1, 
                                        index=relevant_indices.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, output["logits"].size(-1)))
                logits = logits.squeeze(1)
                labels = labels[mask].view(-1)                         
            
            predictions = logits.argmax(dim=1)
            predictions_sum = predictions_sum + predictions.tolist()
            labels_sum  = labels_sum  + labels.tolist()
            
            loss.append(output["loss"].item())
    
    loss =  np.mean(loss)
    return {"predictions":predictions_sum,
            "labels":labels_sum,
            "loss":loss}

def compute_metrics_mutiple(predictions,labels): 
    results = {}   
    preds, label_ids = np.array(predictions),np.array(labels)
    accuracy = accuracy_score(label_ids, preds)
    weighted_f1 = f1_score(label_ids, preds, average='weighted')
    macro_f1 = f1_score(label_ids, preds, average='macro')
    results["accuracy"] = accuracy
    results["weighted_f1"] = weighted_f1
    results["macro_f1"] = macro_f1
    return results

def get_peft_config(g_args,lora_r):
    peft_config = LoraConfig(
                    inference_mode = False,
                    r = lora_r,
                    lora_alpha =  g_args["lora_alpha"],
                    lora_dropout = g_args["lora_dropout"],
                    )
    if "bert" in g_args["model_path"]:
        if g_args["data_format"] in [1,201,202,203,204,3,4]: 
            peft_config.target_modules = ["query", "value"]
        elif g_args["data_format"] in [5,601,602,603,604,7,8]:
            peft_config.target_modules = ["query", "value"]      
    elif ("Qwen" in g_args["model_path"]) or ("MiniCPM" in g_args["model_path"]):
        if g_args["data_format"] in [1,201,202,203,204,3,4]:
            peft_config.target_modules = ["q_proj", "v_proj"]
        elif g_args["data_format"] in [5,601,602,603,604,7,8]:
            peft_config.target_modules = ["q_proj", "v_proj"]
    elif "gte-multilingual" in g_args["model_path"]:
        if g_args["data_format"] in [1,201,202,203,204,3,4]:
            peft_config.target_modules = ["qkv_proj"]
        elif g_args["data_format"] in [5,601,602,603,604,7,8]:
            peft_config.target_modules = ["qkv_proj"]
    return peft_config

def load_model(g_args):
    # Load the model
    if ("bert" in  g_args["model_path"]) or ("gte-multilingual" in g_args["model_path"]):
        if g_args["data_format"] in [1,201,202,203,204,3,4]:
            model = AutoModelForSequenceClassification.from_pretrained(
                pretrained_model_name_or_path=get_abs_path(g_args["model_path"]),
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
                num_labels=g_args["num_labels"]).cuda()
            model.classifier = layer_params_init(model.classifier)
            model.classifier = LayerToFloat32(model.classifier)

        elif g_args["data_format"] in [5,601,602,603,604,7,8]:
            model = AutoModelForMaskedLM.from_pretrained(
                pretrained_model_name_or_path=get_abs_path(g_args["model_path"]),
                torch_dtype=torch.bfloat16,
                trust_remote_code=True).cuda()
        
    elif ("Qwen" in g_args["model_path"]) or ("MiniCPM" in g_args["model_path"]):
        if g_args["data_format"] in [1,201,202,203,204,3,4]:
            model = AutoModelForSequenceClassification.from_pretrained(
                pretrained_model_name_or_path=get_abs_path(g_args["model_path"]),
                torch_dtype=torch.bfloat16,
                device_map = "auto",
                trust_remote_code=True,
                num_labels=g_args["num_labels"]).cuda()
            model.score = layer_params_init(model.score)
            model.score = LayerToFloat32(model.score)

        elif g_args["data_format"] in [5,601,602,603,604,7,8]:
            model = AutoModelForCausalLM.from_pretrained(
                pretrained_model_name_or_path=get_abs_path(g_args["model_path"]),
                torch_dtype=torch.bfloat16,
                device_map = "auto",
                trust_remote_code=True).cuda()
                
    if g_args["peft_config"] is not None:
        peft_model = get_peft_model(model, g_args["peft_config"])
    return peft_model

def prepare_model(model,tuning_stage,g_args): 
    # Training phase 1: only the classification layer is trained
    if tuning_stage%3 == 1:
        for name,param in model.named_parameters():
            param.requires_grad = False
        
        if ("bert" in  g_args["model_path"]) or ("gte-multilingual" in g_args["model_path"]):     
            for name,param in model.classifier.named_parameters():
                param.requires_grad = True
        
        elif ("Qwen" in g_args["model_path"]) or ("MiniCPM" in g_args["model_path"]):
            for name,param in model.score.named_parameters():
                param.requires_grad = True
        
        logging.info(f"tuning_stage ={tuning_stage},The number of parameters is as follows:")
        check_parameters(model)
    
    # Training phase 2: only the pre-trained portion is trained
    elif tuning_stage%3 == 2:
        if g_args["finetune_pattern"] == "lora":
            for name,param in model.named_parameters():
                if "lora" in name:
                    param.data = param.data.to(torch.float32)
                    param.requires_grad = True
                else:
                    param.requires_grad = False
        
        elif g_args["finetune_pattern"] == "full":
            for name,param in model.named_parameters():
                param.requires_grad = True
            if ("bert" in  g_args["model_path"]) or ("gte-multilingual" in g_args["model_path"]):     
                for name,param in model.classifier.named_parameters():
                    param.requires_grad = False
            elif "Qwen" in g_args["model_path"]:
                for name,param in model.score.named_parameters():
                    param.requires_grad = True
        logging.info(f"tuning_stage ={tuning_stage},The number of parameters is as follows:")
        check_parameters(model)
    
    # Training phase 3: both the classification layer and the LoRA layer are trained simultaneously
    elif tuning_stage%3 == 0:
        if g_args["finetune_pattern"] == "lora":
            # Set the LoRA portion to be trainable
            for name,param in model.named_parameters():
                if "lora" in name:
                    param.data = param.data.to(torch.float32)
                    param.requires_grad = True
                else:
                    if ("bert" in  g_args["model_path"]) or ("gte-multilingual" in g_args["model_path"]):
                        if "classifier" in name:
                            param.requires_grad = True
                    elif ("Qwen" in g_args["model_path"]) or ("MiniCPM" in g_args["model_path"]): 
                        if "score" in name:
                            param.requires_grad = True
                    else:
                        param.requires_grad = False
                        
        elif g_args["finetune_pattern"] == "full":
            for name,param in model.named_parameters():
                param.requires_grad = True       
        
        logging.info(f"tuning_stage ={tuning_stage},The number of parameters is as follows:")
        check_parameters(model)
    
    else:
        logging.info("wrong input, tuning_stage should be a positive integer")
    
    return(model)