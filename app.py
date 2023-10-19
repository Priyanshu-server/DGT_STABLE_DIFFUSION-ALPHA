import requests
import os
from PIL import Image
from io import BytesIO
from uuid import uuid4
import pandas as pd
from abc import ABC,abstractmethod
from unsplash import get_unsplash_img_links
from google_search import get_google_img_links
from tqdm import tqdm
import json
from main import main
from data_config import *
from config import USE_DREAM_BOOTH,CLASS_PROMPT,CLASS_DIR

#html5lib must install

class GetImageAbstract(ABC):
    @abstractmethod
    def __init__(self,prompt: str,data_dir: str,prompt_file:str,num_imgs:int,size:int) -> None:
        pass
    @abstractmethod
    def _read_prompt_file(self,prompt_file: str):
        pass
    @abstractmethod
    def run_prompts(self) -> None:
        pass
    @abstractmethod
    def get_image(self,prompt:str,verbose:bool) -> None:
        pass
    @abstractmethod
    def save_img(self,response: requests.Response,name:str,prompt:str) -> None:
        pass
    @abstractmethod
    def resize_with_padding(self,image : Image.Image,target_size:tuple) -> Image.Image:
        pass

class GetImage(GetImageAbstract):
    def __init__(self,prompt = None,data_dir = "data",prompt_file = None,
                 num_imgs = 0,size = None,verbose = True,site = "Google",
                 unsplash_size = None,train = False):
        self.prompt = prompt
        self.prompt_file = prompt_file
        self.num_imgs = num_imgs
        self.size = size
        self.verbose = verbose
        self.dataframe = pd.DataFrame()
        self.data_dir = os.path.join(os.getcwd(), data_dir)
        self.site = site
        self.unsplash_size = unsplash_size if unsplash_size else 'small'
        if not os.path.exists(self.data_dir):
            os.mkdir(self.data_dir)

    def _read_prompt_file(self,prompt_file):
        with open(prompt_file,'rb') as f:
            lines = [line.decode('utf-8').strip() for line in f.readlines()]
            lines = ['+'.join(line.split(" ")) for line in lines]
            print(lines)
        return lines
    
    def save_csv_to_jsonl(self,dataframe):
        output_file = os.path.join(self.data_dir,'metadata.jsonl')

        with open(output_file, 'w') as jsonl_file:
            for _, row in dataframe.iterrows():
                json_obj = {
                    'file_name': row['image'].split("/")[-1],
                    'text': row['text']
                }
                jsonl_file.write(json.dumps(json_obj) + '\n')

    def run_prompts(self):
        if self.prompt_file and self.prompt_file != "":
            prompts = self._read_prompt_file(self.prompt_file)
            for prompt in tqdm(prompts):
                self.get_image(prompt,verbose = self.verbose)
        elif self.prompt:
            self.prompt = '+'.join(self.prompt.split(" "))
            self.get_image(self.prompt,verbose = self.verbose)
        else:
            print("Enter prompt_file or prompt inside args")
        
        ### saving dataframe
        try:
            self.dataframe.to_csv("data.csv",index = False)
            self.save_csv_to_jsonl(self.dataframe)
        except:
            print("Error in saving files !")
        else:
            print("CSV & JSONL Created <-> Saved !")

    # Main Function (all functions -> get_image function helpers)
    def get_image(self,prompt,verbose = True):
        links = None
        if prompt:
            to_search = prompt
            ### if search engine is google
            if self.site == 'google':
                links = get_google_img_links(to_search)
            ### if search engine is unsplash
            elif self.site == 'unsplash':
                links = get_unsplash_img_links(self.num_imgs,size = self.unsplash_size,
                                               prompt = to_search)

            if links:
                counter = 0
                for i,img in enumerate(links):
                    if img.split('.')[-1] == 'gif':
                        if verbose:
                            print('Gif Detected')   
                        continue
                    response = requests.get(img)
                    if response.status_code:
                        self.save_img(response, f"{i}_{str(uuid4())}_img.png",prompt)
                        counter += 1
                    if counter == self.num_imgs:
                        break

        if verbose:
            print("Done !")
    
    def save_img(self,response,name,prompt):
        img = Image.open(BytesIO(response.content)).convert('RGB')

        if self.size and self.site == 'google':
            img = self.resize_with_padding(img,self.size)
        address = os.path.join(self.data_dir, name)
        img.save(address)
        #saving data in dataframe
        if self.dataframe.empty:
            self.dataframe = pd.DataFrame({'image':[address],'text':[prompt.lower()]})
        else:
            self.dataframe = pd.concat([self.dataframe,
                                        pd.DataFrame({'image':[address],'text':[prompt.lower()]})],
                                        ignore_index=True)


    def resize_with_padding(self,image, target_size):
        width, height = image.size
        new_width, new_height = target_size
        aspect_ratio = width / height

        if width > height:
            new_height = int(new_width / aspect_ratio)
        else:
            new_width = int(new_height * aspect_ratio)

        resized_image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        final_image = Image.new("RGB", target_size)
        x = (new_width - width) // 2
        y = (new_height - height) // 2
        final_image.paste(resized_image, (x, y))
        return final_image
    
class GetDreamboothImage(GetImage):
    def __init__(self,prompt = None,data_dir = "data",prompt_file = None,
                 num_imgs = 0,size = None,verbose = True,site = "Google",
                 unsplash_size = None,train = False):
        super().__init__(prompt,data_dir,prompt_file,num_imgs,
                         size,verbose,site,unsplash_size,train)

    def get_image(self,prompt,verbose=True):
        prompt = '+'.join(prompt.split(' '))
        super().get_image(prompt,verbose)

    def get_all_images(self):
        prompt = self.prompt
        verbose = self.verbose
        self.get_image(prompt,verbose)

    def save_img(self,response,name,prompt):
        img = Image.open(BytesIO(response.content)).convert('RGB')

        if self.size and self.site == 'google':
            img = self.resize_with_padding(img,self.size)
        address = os.path.join(self.data_dir, name)
        img.save(address)
    


if __name__ == "__main__":

    ## Triggering Main GetImage Class
    if not USE_DREAM_BOOTH:
        get_image = GetImage(prompt = PROMPT_TEXT, data_dir = DATA_DIR,
                            prompt_file = PROMPT_FILE, num_imgs = NUM_IMGS,
                            site = SITE, size = SIZE,
                            train = TRAIN)
        get_image.run_prompts()
        
    else:
        ## If using Dreambooth
        if not PROMPT_TEXT or PROMPT_TEXT == "":
            raise ValueError("Prompt Required for Dreambooth Data Creation")
        
        get_image = GetDreamboothImage(prompt = PROMPT_TEXT, data_dir = DATA_DIR,
                            prompt_file = "", num_imgs = NUM_IMGS,
                            site = SITE, size = SIZE,train = TRAIN)
        get_image.get_all_images()

    if TRAIN:
        main()
    
    # get_image = GetDreamboothImage()
    # get_image.get_all_images()

