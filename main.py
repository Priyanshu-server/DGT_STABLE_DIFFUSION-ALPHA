import os
from lx_cmd import (install_deps,diffuser_clone,
                    install_diffuser,set_acc_precision_fp16)
from config import *


def main():
    os.system(command=install_deps.cmd)
    if not os.path.exists("diffusers/"):
        os.system(command=diffuser_clone.cmd)
    os.system(command=install_diffuser.cmd)
    os.system(command=set_acc_precision_fp16.cmd)

    from huggingface_hub import login
    login(ACCESS_TOKEN)

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

    accelerator_cmd += '--resume_from_checkpoint' if RESUME_FROM_CHECKPOINT else ""
    if TRAIN_DATA_DIR == "" and DATASET_NAME:
        accelerator_cmd += f' --dataset_name={DATASET_NAME}'
    elif DATASET_NAME == "" and TRAIN_DATA_DIR:
        accelerator_cmd += f' --train_data_dir={TRAIN_DATA_DIR}'
    else:
        print("Something is worng in file locations ! Check Readme.md")
        exit()
    print("\n\n\n\n\n",accelerator_cmd,"\n\n\n\n\n")


    os.system(accelerator_cmd)

if __name__ == "__main__":
    main()

