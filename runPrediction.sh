
docker run -v `pwd`/data:/data -v `pwd`/src:/scr wlo-cuda:1.0 /usr/bin/python3 /scr/predict.py /data/wirlernenonline.oeh-embed.h5 /data/wirlernenonline.oeh-id.pickle "$1"