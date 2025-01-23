architecture=ST-Net
backbone=densenet121
loss_main=RMSE
lambda_main=0.
lambda_rec=0.

optim_fn=sgd
lr_rate=0.01
max_epochs=50
batch_size=32
scheduler_fn=NoAdjust # CosineAnnealingLR


exp_code=${architecture}_${backbone}_${loss_main}_a${lambda_main}b${lambda_rec}_ep${max_epochs}_bs${batch_size}_${optim_fn}lr${lr_rate}_${scheduler_fn}


CUDA_VISIBLE_DEVICES=0 python main.py -c cfgs/cfgs_lung.yaml \
--opts CKPTS exp_code ${exp_code} HyperParams optim_fn ${optim_fn} HyperParams scheduler_fn ${scheduler_fn} HyperParams max_epochs ${max_epochs} HyperParams batch_size ${batch_size} HyperParams loss_main ${loss_main} HyperParams lambda_main ${lambda_main} HyperParams lambda_rec ${lambda_rec} HyperParams architecture ${architecture} HyperParams backbone ${backbone} HyperParams lr_rate ${lr_rate}


CUDA_VISIBLE_DEVICES=0 python main.py -c cfgs/cfgs_her2st.yaml \
--opts CKPTS exp_code ${exp_code} HyperParams optim_fn ${optim_fn} HyperParams scheduler_fn ${scheduler_fn} HyperParams max_epochs ${max_epochs} HyperParams batch_size ${batch_size} HyperParams loss_main ${loss_main} HyperParams lambda_main ${lambda_main} HyperParams lambda_rec ${lambda_rec} HyperParams architecture ${architecture} HyperParams backbone ${backbone} HyperParams lr_rate ${lr_rate}


CUDA_VISIBLE_DEVICES=0 python main.py -c cfgs/cfgs_stnet.yaml \
--opts CKPTS exp_code ${exp_code} HyperParams optim_fn ${optim_fn} HyperParams scheduler_fn ${scheduler_fn} HyperParams max_epochs ${max_epochs} HyperParams batch_size ${batch_size} HyperParams loss_main ${loss_main} HyperParams lambda_main ${lambda_main} HyperParams lambda_rec ${lambda_rec} HyperParams architecture ${architecture} HyperParams backbone ${backbone} HyperParams lr_rate ${lr_rate}

# independent test
CUDA_VISIBLE_DEVICES=0 python test_evaluation.py -ep results/humanlung_cell2location/${exp_code}
CUDA_VISIBLE_DEVICES=0 python test_evaluation.py -ep results/her2st/${exp_code}
CUDA_VISIBLE_DEVICES=0 python test_evaluation.py -ep results/stnet/${exp_code}