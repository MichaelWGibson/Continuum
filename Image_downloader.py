import nucleus
import requests
import pprint as pp
import shutil
import sys
import json
from tqdm import tqdm

usage_str = '''Usage:
    python ./image_downloader.py api_key mode data_id option

    api_key: Your live Scale API key
    mode: "d" for dataset, "s" for slice  ## CURRENTLY NOT IMPLEMENTED ##
    data_id: dataset or slice id      ## CURRENTLY NOT IMPLEMENTED ##
    option: "images" for image downloads, "annotations" for annotation downloads

'''

def main():
    if len(sys.argv) != 5:
        print(usage_str)
        sys.exit(1)

    api_key = sys.argv[1]
    #mode = sys.argv[2]
    #dataset = sys.argv[3]
    option = sys.argv[4]
    
    #"live_57744f35ea1e47d4b3e6f0d39f1e080c"

    client = nucleus.NucleusClient(api_key)


    if option == "images":
        for i in tqdm(range(0, 1000, 100), desc="Downloading..."):
            url = "https://api.scale.com//v1/nucleus/dataset/ds_bzvafq6bcapg1q6d79qg/iloc/%s?format=image" % str(i)

            response = requests.get(
                url,
                headers={"Content-Type": "application/json"},
                auth=(api_key, ""),
            )

            path = "images/img%s.jpg" % str(i)
            with open(path, 'wb') as f:
                    for chunk in response:
                        f.write(chunk)

    if option == "annotations":
        for i in tqdm(range(0, 1000, 100), desc="Downloading..."):
            url = "https://api.scale.com//v1/nucleus/dataset/ds_bzvafq6bcapg1q6d79qg/iloc/%s" % str(i)

            response = requests.get(
                url,
                headers={"Content-Type": "application/json"},
                auth=(api_key, ""),
            )

            path = "annotations/img%s.json" % str(i)
            with open(path, 'wb') as f:
                f.write(response.content)



if __name__ == "__main__":
    main()
