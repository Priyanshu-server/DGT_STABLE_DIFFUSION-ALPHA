import requests
from PIL import Image
from io import BytesIO
from config import UNSPLASH_ACCESS_TOKEN

access_token  = UNSPLASH_ACCESS_TOKEN
autherization_header = f"Client-ID {access_token}"
headers = {'Authorization':autherization_header}


def get_unsplash_img_links(num_imgs,size,prompt):
    API = f"https://api.unsplash.com/search/photos?page=1&query={prompt}"
    response = requests.get(API,headers=headers)
    response = response.json()
    links = []
    for img in response['results'][:num_imgs]:
        links.append(img['urls'][size])
    return links    

if __name__ == '__main__':
    links = get_unsplash_img_links(3,'small','cat+dog')
    print(links)