ACCESS_TOKEN = "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"
UNSPLASH_ACCESS_TOKEN = "GVC_0N1u-vKofzEFj38pWYtIF1F69aK7nZb90oi_vK0"

# ACCELERATE
SET_ACCELERATE = 'default'

# DICTS
MODEL_NAME = 'OFA-Sys/small-stable-diffusion-v0'
DATASET_NAME = 'lambdalabs/pokemon-blip-captions'
# DATASET_NAME = ''
OUTPUT_DIR = 'OFAsd-pokemon-model'

#Image
RESOLUTION = 512

#Gradient
GRADIENT_ACCUMULATION_STEPS = 4
GRADIENT_CHECKPOINTING = True
MIXED_PRECISION = 'no' #[fp16,bf16]

#Training ARGS
TRAIN_BATCH_SIZE = 1
MAX_TRAIN_STEPS = 100
LEARNING_RATE = 1e-05
MAX_GRAD_NORM = 1
LR_SCHEDULER = "constant"
LR_WARMUP_STEPS = 0
TRAIN_DATA_DIR = ''
TRAIN_DATA_DIR = 'train/'


#Checkpointing

CHECKPOINTING_STEPS = 200
CHECKPOINTS_TOTAL_LIMIT = 5 
RESUME_FROM_CHECKPOINT = ""

#OPTIMIZATION
## Do not use if sudo if not allowed or the GPU have enough RAM
### Only use for large models with <=16GiB GPU
USE_8BIT_ADAM = False 

# DreamBooth Configs
USE_DREAM_BOOTH = True
CLASS_DIR = "class_dir"
CLASS_PROMPT = "class prompt"
INSTANCE_PROMPT = "instance prompt"
INSTANCE_DICT = "train/"
NUM_CLASS_IMGS = 5