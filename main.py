# Configure the number of GPUs
import sys,os
py_dir = os.path.dirname(os.path.realpath(__file__))
utils_path = os.path.abspath(os.path.join(py_dir, "utils"))
sys.path.append(utils_path)
import init_utils

import json,time,logging,argparse
from datetime import datetime
from transformers import AutoTokenizer

from general_settings import setup_logging,ConfigManager
from model_proc import get_peft_config
from train import pipeline_tuning
from basic_function import get_abs_path

def main():   
    # Set log information 
    setup_logging()
    start_time = datetime.fromtimestamp(time.time())
    logging.info(f"start_time:{start_time.strftime('%Y-%m-%d %H:%M:%S')}")
       
    # The path to the config file should be provided when running the main function
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path",type = str, default = get_abs_path("../configs/training_config.json"))
    args = parser.parse_args()
    config_path = args.config_path
    
    # Get information from the config
    config_manager = ConfigManager(config_path)
    g_args = config_manager.get_full_config()
    for key,value in g_args.items():
        logging.info(f"{key}:{value}")
    
    # Supplement the tokenizer for g_args
    g_args["tokenizer"] = AutoTokenizer.from_pretrained(get_abs_path(g_args["model_path"]),trust_remote_code=True)
    if ("Qwen" in g_args["model_path"]) or ("MiniCPM" in g_args["model_path"]):
        g_args["tokenizer"].pad_token = g_args["tokenizer"].eos_token
        g_args["tokenizer"].pad_token_id = g_args["tokenizer"].eos_token_id
        if "Qwen" in g_args["model_path"]:
            g_args["tokenizer"].mask_token_id = 81535
            g_args["tokenizer"].mask_token = 'masked'
        elif "MiniCPM" in g_args["model_path"]:
            g_args["tokenizer"].mask_token_id = 17801
            g_args["tokenizer"].mask_token = 'mask'
    # Each setting (a combination of r, batch_size, and learning_rate) calls the pipeline_tuning function once and returns the result for that setting
    result = {}      
    for lora_r in g_args["lora_rs"]:
        for batchsize in g_args["batchsizes"]:        
            for lr in g_args["lrs"]: 
                if g_args["finetune_pattern"] == "lora": 
                    peft_config = get_peft_config(g_args,lora_r)
                    g_args["peft_config"] = peft_config
                    setting = f"lora_r_{lora_r}_bs_{batchsize}_slr_{lr}"
                elif  g_args["finetune_pattern"] == "full":
                    g_args["peft_config"] = None
                    setting = f"full_finetune_bs_{batchsize}_slr_{lr}"                       
                result_setting = pipeline_tuning(g_args=g_args,
                                                lora_r=lora_r,
                                                batchsize=batchsize,
                                                lr=lr)
                logging.info(json.dumps(result_setting))
                logging.info(30*"^"+"setting分界"+30*"^")         
                result[setting] = result_setting
        
    end_time = datetime.fromtimestamp(time.time())
    elapsed_time = end_time - start_time
    hours, remainder = divmod(elapsed_time.total_seconds(), 3600)
    minutes, seconds = divmod(remainder, 60)
    logging.info(f"end_time:{end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    logging.info(f"Elapsed time: {int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}")

if __name__ == "__main__":
    main()