import torch,logging,datetime,json,re
from basic_function import get_abs_path

# Configure the logging
def setup_logging():
    start_time = datetime.datetime.now()
    file_name = f'../../log/{start_time.strftime("%Y_%m_%d_%H_%M_%S")}.log'     
    file_name = get_abs_path(file_name)
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    
    file_handler = logging.FileHandler(file_name,encoding='utf-8')
    file_handler.setFormatter(logging.Formatter('%(message)s'))
    
    adapter_logger = logging.getLogger("peft.tuners.tuners_utils")
    adapter_logger.setLevel(logging.WARNING)
       
    logger = logging.getLogger()
    if not hasattr(logger, 'file_handler_added'):
        logger.addHandler(file_handler)
        logger.file_handler_added = True

# Get the number of GPUs in the system        
def generate_device_list():
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
    else:
        device_count = 0
        
    if device_count == 0:
        return ""  
    devices = list(range(device_count))   
    return ",".join(map(str, devices))

class ConfigManager:
    '''
    A class for managing training configurations of different models
    '''
    def __init__(self, config_path):
        self.config_path = config_path
        self.config = {}
        with open(self.config_path, 'r',encoding='utf-8') as f:
            raw_json_text = f.read()
        json_text_without_comments = self.remove_comments(raw_json_text)
        try:
            self.config = json.loads(json_text_without_comments)
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON: {e}")

    def remove_comments(self, text):
        # Remove C-style comments
        text = re.sub(r'//.*', '', text)
        # Remove Python-style comments
        text = re.sub(r'#.*', '', text)
        # Remove block comments
        text = re.sub(r'/\*.*?\*/', '', text, flags=re.DOTALL)
        return text
    
    def get_full_config(self):
        return self.config
