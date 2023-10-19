import os
from lx_cmd import (install_deps,diffuser_clone,
                    install_diffuser,set_acc,set_acc__default_precision_fp16)
from config import *


def main():
    os.system(command=install_deps.cmd)
    from gen_accelerator_cmd import gen_accelerator_text_img,gen_accelerator_dreambooth
    if not os.path.exists("diffusers/"):
        os.system(command=diffuser_clone.cmd)
    os.system(command=install_diffuser.cmd)

    # Accelerator Config
    if SET_ACCELERATE.lower() == 'default':
        os.system(command=set_acc__default_precision_fp16.cmd)
    else:
        os.system(command=set_acc.cmd)

    from huggingface_hub import login
    login(ACCESS_TOKEN)

    if not USE_DREAM_BOOTH:
        accelerator_cmd = gen_accelerator_text_img()
    else:
        accelerator_cmd = gen_accelerator_dreambooth()
    os.system(accelerator_cmd)

if __name__ == "__main__":
    main()

