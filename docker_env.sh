docker run -dit \
    --name cuca_dev_cyyan \
    --gpus all \
    --cpus 64 \
    --mount type=bind,source=/home/cyyan/Projects/CUCA/,target=/home/appuser/CUCA \
    --shm-size 128g \
    -p 6009:6009 \
    charing/hest_dev:cuda12.1 \
    /bin/bash -c 'while true; do echo `date`; sleep 600; done'

echo "docker exec -it cuca_dev_cyyan /bin/bash"
# docker stop cuca_dev_cyyan
# docker rm cuca_dev_cyyan
# docker exec -it --user root cuca_dev_cyyan /bin/bash

#    --mount type=bind,source=/usr/lib/x86_64-linux-gnu/,target=/usr/lib/x86_64-linux-gnu/ \
