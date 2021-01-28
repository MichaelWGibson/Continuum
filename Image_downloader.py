import nucleus
import requests
import pprint as pp
import shutil
import sys
import json

def main(api_key, dataset, option):
    api_key = api_key
    
    #"live_57744f35ea1e47d4b3e6f0d39f1e080c"

    client = nucleus.NucleusClient(api_key)

    #dataset = dataset
    dataset = client.get_dataset("ds_bzvafq6bcapg1q6d79qg")

    if option == "images":
        for i in range(0, 1000, 100):
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
        for i in range(0, 1000, 100):
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
    main(sys.argv[1], sys.argv[2], sys.argv[3])
