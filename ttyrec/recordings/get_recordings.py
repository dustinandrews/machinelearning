"""
Download nethack recordings
"""

from bs4 import BeautifulSoup, SoupStrainer
import hashlib
import os
from urllib.request import urlopen

class ttyrec_download():
    
    # f = urllib.request.urlopen("http://stackoverflow.com")
    # print(f.read())
    def __init__(self):
        self.cache_dir = "./cache"
        self.recording_dir = "./recordings"
        self.player_url = "https://alt.org/nethack/topallclassplayers.html"
        self.base_url = "https://alt.org/nethack/"
    
    def get_hash(self, string):
        hash_function =  hashlib.sha224()
        hash_function.update(string.encode())
        return hash_function.digest().hex()
    
    def get_file_path(self, url):
        f_hash = self.get_hash(url)
        f_name = self.cache_dir + "/" + f_hash + ".html"
        return f_name
    
    def get_page(self, url):
        local_copy = self.check_cache(url)
        if local_copy == None:            
            response = urlopen(url)
            local_copy = response.read().decode('utf-8')
            self.cache_page(url, local_copy)
        if str(local_copy[0]).startswith(r'<!DOCTYPE'): #strip doctype
            local_copy = local_copy[1:] 
        return local_copy
            
    
    def check_cache(self, url):
        if os.path.exists(self.cache_dir):
            f_name = self.get_file_path(url)
            if os.path.exists(f_name):
                with open(f_name, "r") as f:
                    return f.read()
        else:
            os.mkdir(self.cache_dir)
            return None
        
    def cache_page(self, url, data):
        f_name = self.get_file_path(url)
        with open(f_name, "w") as f:
            f.write(data)
        
    def get_player_urls(self):
        player_page_html = self.get_page(self.player_url)
        cache_key = 'player urls'
        player_urls = self.check_cache(cache_key)
        if player_urls == None:
            strainer = SoupStrainer('a')
            soup = BeautifulSoup(str(player_page_html), 'lxml', parse_only=strainer)
            player_urls = [h['href'] for h in soup.find_all('a', href=True) if h['href'].startswith('plr')]
            self.cache_page(cache_key, "\n".join(player_urls))
        else:
            player_urls = player_urls.split("\n")

        return player_urls
    
    
    def get_player_dir(self, name):
        return self.recording_dir + "/" + name
        
    def get_player_ttyrec_index(self, player_name):        
        ttypage = 'browsettyrec.php?player=' + player_name
        url = self.base_url + ttypage
        print(url)
        page = self.get_page(url)            
        player_dir = self.get_player_dir(player_name)
        os.makedirs(player_dir, exist_ok=True)
        with open(player_dir + "/browsettyrec.html" , "w") as p:
                p.write(page)
        return self.parse_tty_list_page(page)

    def parse_tty_list_page(self, page):
        strainer = SoupStrainer('a')
        soup = BeautifulSoup(str(page), 'lxml', parse_only=strainer)
        ttyrecs = [h['href'] for h in soup.find_all('a', href=True) if h['href'].startswith('https://s3.amazonaws.com/altorg/ttyrec/')]
        return ttyrecs
     
    def download_binary(self, url, local_dir):
        file_name = local_dir + "/" + url.split('/')[-1].replace(":", "_")
        if not os.path.exists(file_name):
            u = urlopen(url)
            f = open(file_name, 'wb')
            meta = u.info()            
            file_size = int(meta["Content-Length"])
            print( "Downloading: {} Bytes: {}".format(file_name, file_size))
            
            file_size_dl = 0
            block_sz = 8192
            while True:
                buffer = u.read(block_sz)
                if not buffer:
                    break
            
                file_size_dl += len(buffer)
                f.write(buffer)
                status = r"%10d  [%3.2f%%]" % (file_size_dl, file_size_dl * 100. / file_size)
                status = status + chr(8)*(len(status)+1)
                print (status, end="")       
            f.close()
        else:
            print("Already have {}".format(file_name))
        


if __name__ == '__main__':
    dl = ttyrec_download()
    urls = dl.get_player_urls()
    player_url = urls[0]
    
    name = player_url.split("=")[1]
    tty_index = dl.get_player_ttyrec_index(name)
    for index in tty_index:
        dl.download_binary(index, dl.get_player_dir(name))
    
    