from dataclasses import dataclass

@dataclass
class Command:
    name:str
    cmd:str
    help:str

install_deps = Command(name="Dependies Installation",
                       cmd = "pip3 install datasets==2.14.5 transformers==4.34.0 accelerate==0.23.0 torchvision==0.16.0 ftfy==6.1.1 tensorboard==2.14.1",
                       help = "It helps to install the required dependies in the Environemnt")
                       #torch version depends upon hardware to utilize GPU
install_deps_colab = Command(name="Dependies Installation",
                       cmd = "pip3 install datasets transformers accelerate torchvision ftfy tensorboard",
                       help = "It helps to install the required dependies in the Environemnt")

diffuser_clone = Command(name = "Clone Diffuser",
                         cmd = "git clone https://github.com/huggingface/diffusers.git",
                         help = "It provides all the necessary files required to fine tune Models")

install_diffuser = Command(name = "Diffuser Installation",
                           cmd = "pip3 install ./diffusers",
                           help = "Install the diffuser library from diffuser_clone")
set_acc = Command(name = "Accelerate Precision fp16",
                                 cmd = "accelerate config",
                                 help = "It helps to initialize the accelerate config file and put precision to fp16")


set_acc__default_precision_fp16 = Command(name = "Accelerate Precision fp16",
                                 cmd = "accelerate config default --mixed_precision fp16",
                                 help = "It helps to initialize the accelerate config file and put precision to fp16")
                                #  cmd = "accelerate config default --mixed_precision fp16",

install_bitsandbytes = Command(name = "Install BitsAndBytes for 8bit optimization",
                               cmd = "pip3 install bitsandbytes",
                               help = "In given sudo permission and cuda installed it will help in 8bit optimization")
