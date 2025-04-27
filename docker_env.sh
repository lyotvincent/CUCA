projname=CUCA
abspath=/home/cyyan/Projects/
imagename=crpi-vgk1sxg5ba2a8sff.cn-hangzhou.personal.cr.aliyuncs.com/cuca_proj_env/cuca_env

docker run -dit \
    --name cuca_proj_env \
    --gpus all \
    --cpus 64 \
    --mount type=bind,source=${abspath}${projname},target=/home/appuser/${projname} \
    --shm-size 128g \
    -p 6009:6009 \
    ${imagename}:v1_0 \
    /bin/bash -c 'while true; do echo `date`; sleep 600; done'

echo "docker exec -it cuca_proj_env /bin/bash"
# docker stop cuca_proj_env
# docker rm cuca_proj_env
# docker exec -it --user root cuca_proj_env /bin/bash

#    --mount type=bind,source=/usr/lib/x86_64-linux-gnu/,target=/usr/lib/x86_64-linux-gnu/ \
