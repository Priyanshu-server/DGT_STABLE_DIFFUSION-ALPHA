import requests
from bs4 import BeautifulSoup

def get_google_img_links(to_search):
    URL = f'https://www.google.com/search?q={to_search.replace(" ","+")}&sca_esv=572530057&tbm=isch&sxsrf=AM9HkKk4xSL5p0B86w0c6J8U7vTo4uwZyQ:1697029121365&source=lnms&sa=X&ved=2ahUKEwj5-s6Phu6BAxVJ1TgGHeYFDIoQ_AUoAXoECAIQAw&cshid=1697029147107705&biw=1296&bih=654&dpr=1'
    r = requests.get(URL)
    soup = BeautifulSoup(r.content, 'html5lib') # If this line causes an error, run 'pip install html5lib' or install html5lib 
    images = soup.find_all("img")
    images = [img['src'] for img in images]
    return images

if __name__ == "__main__":
    links = get_google_img_links(to_search = "cat+dog")
    print(links)