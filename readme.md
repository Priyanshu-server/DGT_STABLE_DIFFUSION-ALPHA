## README FILE
### Inspired By HuggingFace
![Hugging Face Image](https://huggingface.co/front/assets/homepage/hugs-mobile.svg)

### Tokens
>**ADD ACCESS_TOKEN & UNSPLASH_ACCESS_TOKEN in `config.py`**
>**If you are using the `predownloaded Dataset | HuggingFace Dataset` then avoid adding `UNSPLASH_ACCESS_TOKEN`**
>**`ACCESS_TOKEN` is the token from HuggingFace, so it can download the Hugging Face Dataset | HuggingFace Models**

## Requirements
- Data Generation Requirements are inside requirements.txt `pip3 install -r requirements.txt`

### Config.py
- Update config.py file for specific changes - 
    - Model Name
    - Dataset Address
    - ACCESS Token
    - etc.
    - 
### Main.py File
- It is the driver file, execute only Main.py file directly to train the Stable Diffusion Model

### app.py File
- It is file to download data, but if you pass the arg - train True then it will automatically trigger main.py file in order to train the model on the downloaded data.

### Things to keep in mind
- Working with online data (COMMENT TRAIN_DATA_DIR) and DATASET_NAME = 'ONLINE_DATASET_NAME'
- Working with offline data (COMMENT DATASET_NAME) and TRAIN_DATA_DIR = 'data_dir/'
    - Format -> Data_dir/
        - IMG1
        - IMG2
        - etc.
        - metadata.jsonl
            - {'file_name':'IMG1.png","text":"prompt"}
            - {'file_name':'IMG2.png","text":"prompt"}
            
## Process
### Data Collection
- HuggingFace Dataset
- Online
- Offline
- 
#### HuggingFace DATASET
- Working with HuggingFace dataset not required triggering app.py. Only run `python3 main.py`
- But before execution, update config.py file
    - Change value of TRAIN_DATA_DIR = '' & DATASET_NAME = 'HUGGINGFACE_DATA'

#### ONLINE DATASET
- Working with offline dataset require triggering main.py.
- But before exectuion, update config.py file
    - Change value of TRAIN_DATA_DIR = '/data_dir/' and DATASET_NAME = ''
- Command when downloading pictures from unsplash (Update ACESS TOKEN IN CONFIG.PY)
    - `python3 app.py -prompt_file prompts.txt -num_imgs 2 -size 128 -data_dir train -site unsplash -unsplash_size raw -verbose False -train False`
- unsplash_sized = `{"raw","full","regular","small","thumb"}`
- Command when downloading dataset from Google (No need to provide UNSPLASH ACCESS TOKEN)
    - `python3 app.py -prompt_file prompts.txt -num_imgs 2 -size 128 -data_dir train -verbose False -train False`

#### Offline Dataset
- Working with offline dataset require triggering main.py.
- But before exectuion, update config.py file
    - Change value of TRAIN_DATA_DIR = '/data_dir/' and DATASET_NAME = ''
    - Make a file `metadata.json` inside /data_dir/ and inside that file, each line represents a json object of `["file_name","text"]` keys. The model will read files from that `jsonl : File` only.
    - Put your images inside `/data_dir/` with file `metadata.jsonl`

### Training Model
- Put your HuggingFace Model name inside `MODEL_NAME = ` var (Config.py).
- Change you Dataset pointer and files according to above instructions.
- If using the `app.py` file to build & download dataset then we can pass value `yes | no` value inside `-train` argument. It will automatically trigger `main.py` file after installing the Dataset
- If the data is `pre-downloaded` or `HuggingFace Dataset` then simply edit `config.py` file accordingly and hit `python3 main.py`. It will start model training.
- `OUTPUT_DIR/` is the variable in `config.py` file which can be modified in order to change the `checkpoint & saved model` 

### Core Command
`accelerate launch diffusers/examples/text_to_image/train_text_to_image.py \
          --pretrained_model_name_or_path=$MODEL_NAME \
          --dataset_name=$DATASET_NAME \
          --use_ema \
          --resolution=512 --center_crop --random_flip \
          --train_batch_size=1 \
          --gradient_accumulation_steps=4 \
          --gradient_checkpointing \
          --mixed_precision="no" \
          --max_train_steps=100 \
          --learning_rate=1e-05 \
          --max_grad_norm=1 \
          --lr_scheduler="constant" \
          --lr_warmup_steps=0 \
          --output_dir=$OUTPUT_DIR`