## README FILE

### Inspired By HuggingFace
![Hugging Face Image](https://huggingface.co/front/assets/homepage/hugs-mobile.svg)

### Config.py
- Update config.py file for specific changes - 
    - Model Name
    - Dataset Address
    - ACCESS Token
    - etc.

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

### ONLINE DATASET
- Working with online dataset not require triggering main. Only run `python3 main.py`
- But before execution, update config.py file
    - Change value of TRAIN_DATA_DIR = '' & DATASET_NAME = 'HUGGINGFACE_DATA'

## OFFLINE DATASET
- Working with offline dataset require triggering main.
- But before exectuion, update config.py file
    - Change value of TRAIN_DATA_DIR = '/data_dir/' and DATASET_NAME = ''
- Command when downloading pictures from unsplash (Update ACESS TOKEN IN CONFIG.PY)
    - `python3 app.py -prompt_file prompts.txt -num_imgs 2 -size 128 -data_dir train -site unsplash -unsplash_size raw -verbose False -train False`
- unsplash_sized = `{"raw","full","regular","small","thumb"}`
- Command when downloading dataset from Google (No need to provide UNSPLASH ACCESS TOKEN)
    - `python3 app.py -prompt_file prompts.txt -num_imgs 2 -size 128 -data_dir train -verbose False -train False`




update usplash_token in unsplash file whenever using it


### Core Command
'accelerate launch diffusers/examples/text_to_image/train_text_to_image.py \
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
          --output_dir=$OUTPUT_DIR'