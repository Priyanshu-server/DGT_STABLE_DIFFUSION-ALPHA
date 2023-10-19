from config import *
from lx_cmd import install_bitsandbytes
import os
from dreambooth_huggingface_data import get_dreambooth_data_hugging_face


def gen_accelerator_text_img():
    # --dataset_name={DATASET_NAME} \
    accelerator_cmd = f'accelerate launch diffusers/examples/text_to_image/train_text_to_image.py \
            --pretrained_model_name_or_path={MODEL_NAME} \
            --use_ema \
            --resolution={RESOLUTION} --center_crop --random_flip \
            --train_batch_size={TRAIN_BATCH_SIZE} \
            --gradient_accumulation_steps={GRADIENT_ACCUMULATION_STEPS} '

    if GRADIENT_CHECKPOINTING:
        accelerator_cmd += '--gradient_checkpointing'

    accelerator_cmd += f' --mixed_precision={MIXED_PRECISION} \
                        --max_train_steps={MAX_TRAIN_STEPS} \
                        --learning_rate={LEARNING_RATE} \
                        --max_grad_norm={MAX_GRAD_NORM} \
                        --lr_scheduler={LR_SCHEDULER} \
                        --lr_warmup_steps={LR_WARMUP_STEPS} \
                        --output_dir={OUTPUT_DIR} \
                        --checkpointing_steps={CHECKPOINTING_STEPS} \
                        --checkpoints_total_limit={CHECKPOINTS_TOTAL_LIMIT} '

    if RESUME_FROM_CHECKPOINT != "":
        accelerator_cmd += ' --resume_from_checkpoint={RESUME_FROM_CHECKPOINT}'
    
    if USE_8BIT_ADAM:
        try:
            os.system(command=install_bitsandbytes.cmd)
            accelerator_cmd += " --use_8bit_adam"
        except:
            print("Something Went wrong while installing bitsandbytes")
            print("For more information execute `python -m bitsandbytes`")

    if TRAIN_DATA_DIR == "" and DATASET_NAME != "":
        accelerator_cmd += f' --dataset_name={DATASET_NAME}'
    elif DATASET_NAME == "" and TRAIN_DATA_DIR != "":
        accelerator_cmd += f' --train_data_dir={TRAIN_DATA_DIR}'
    else:
        print("Something is wrong in file locations ! Check Readme.md")
        exit(0)
    print("\n\n\n\n\n",accelerator_cmd,"\n\n\n\n\n",sep = "\n")

    return accelerator_cmd

def gen_accelerator_dreambooth():
    # checking variables
    if INSTANCE_DICT == "" : raise ValueError
    if not os.path.exists(INSTANCE_DICT): raise NotADirectoryError("Make a directory with the same name")

    # Getting Data
    if len(os.listdir(INSTANCE_DICT)) == 0:
        if DATASET_NAME != "":
            try:
                get_dreambooth_data_hugging_face(DATASET_NAME,INSTANCE_DICT)
            except:
                raise LookupError
        else:
            raise ValueError
        
    accelerate_cmd = f"accelerate launch ./diffusers/examples/dreambooth/train_dreambooth.py \
                    --pretrained_model_name_or_path={MODEL_NAME}  \
                    --instance_data_dir={INSTANCE_DICT} \
                    --output_dir=f{OUTPUT_DIR} \
                    --instance_prompt='{INSTANCE_PROMPT}' \
                    --class_prompt='{CLASS_PROMPT}' \
                    --resolution={RESOLUTION} \
                    --train_batch_size={TRAIN_BATCH_SIZE} \
                    --gradient_accumulation_steps={GRADIENT_ACCUMULATION_STEPS} \
                    --learning_rate={LEARNING_RATE} \
                    --lr_scheduler={LR_SCHEDULER} \
                    --lr_warmup_steps={LR_WARMUP_STEPS} \
                    --num_class_images={NUM_CLASS_IMGS} \
                    --max_train_steps={MAX_TRAIN_STEPS} \
                    --class_data_dir={CLASS_DIR} "
    
    if WITH_PRIOR_PRESERVATION:
         if CLASS_DIR == "" or CLASS_PROMPT == "": raise ValueError("CLASS Args in Config.py \
                                                                     should be Valid for prior preservation")
         accelerate_cmd += f" --with_prior_preservation --prior_loss_weight={PRIOR_LOSS_WEIGHT}"
        
    if RESUME_FROM_CHECKPOINT != "":
            accelerate_cmd += ' --resume_from_checkpoint={RESUME_FROM_CHECKPOINT}'
    
    if USE_8BIT_ADAM:
            try:
                os.system(command=install_bitsandbytes.cmd)
                accelerate_cmd += " --use_8bit_adam "
            except:
                print("Something Went wrong while installing bitsandbytes")
                print("For more information execute `python -m bitsandbytes`")

    if GRADIENT_CHECKPOINTING:
            accelerate_cmd += ' --gradient_checkpointing'

    print("\n\n\n\n\n",accelerate_cmd,"\n\n\n\n\n")

    return accelerate_cmd
  
