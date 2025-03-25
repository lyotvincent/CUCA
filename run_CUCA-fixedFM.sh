architecture=CUCAMLP
backbone=virchow2
loss_main=RMSE 

lr_rate=0.002
max_epochs=100
batch_size=128
pre_extracted=1

device=0

lambda_main=0.4
lambda_rec=0.5
exp_code=${architecture}_${backbone}_BN_infoNCE${loss_main}_a${lambda_main}b${lambda_rec}_ep${max_epochs}_bs${batch_size}_lr${lr_rate}_${pre_extracted}

CUDA_VISIBLE_DEVICES=${device} python main.py -c cfgs/cfgs_lung.yaml \
--opts CKPTS exp_code ${exp_code} HyperParams pre_extracted ${pre_extracted} HyperParams max_epochs ${max_epochs} HyperParams batch_size ${batch_size} HyperParams loss_main ${loss_main} HyperParams lambda_main ${lambda_main} HyperParams lambda_rec ${lambda_rec} HyperParams architecture ${architecture} HyperParams backbone ${backbone} HyperParams lr_rate ${lr_rate}
CUDA_VISIBLE_DEVICES=${device} python test_evaluation.py -ep results/humanlung_cell2location/${exp_code}

lambda_main=0.7
lambda_rec=0.2
exp_code=${architecture}_${backbone}_BN_infoNCE${loss_main}_a${lambda_main}b${lambda_rec}_ep${max_epochs}_bs${batch_size}_lr${lr_rate}_${pre_extracted}

CUDA_VISIBLE_DEVICES=${device} python main.py -c cfgs/cfgs_her2st.yaml \
--opts CKPTS exp_code ${exp_code} HyperParams pre_extracted ${pre_extracted} HyperParams max_epochs ${max_epochs} HyperParams batch_size ${batch_size} HyperParams loss_main ${loss_main} HyperParams lambda_main ${lambda_main} HyperParams lambda_rec ${lambda_rec} HyperParams architecture ${architecture} HyperParams backbone ${backbone} HyperParams lr_rate ${lr_rate}
CUDA_VISIBLE_DEVICES=${device} python test_evaluation.py -ep results/her2st/${exp_code}

lambda_main=0.3
lambda_rec=0.6
exp_code=${architecture}_${backbone}_BN_infoNCE${loss_main}_a${lambda_main}b${lambda_rec}_ep${max_epochs}_bs${batch_size}_lr${lr_rate}_${pre_extracted}

CUDA_VISIBLE_DEVICES=${device} python main.py -c cfgs/cfgs_stnet.yaml \
--opts CKPTS exp_code ${exp_code} HyperParams pre_extracted ${pre_extracted} HyperParams max_epochs ${max_epochs} HyperParams batch_size ${batch_size} HyperParams loss_main ${loss_main} HyperParams lambda_main ${lambda_main} HyperParams lambda_rec ${lambda_rec} HyperParams architecture ${architecture} HyperParams backbone ${backbone} HyperParams lr_rate ${lr_rate}
CUDA_VISIBLE_DEVICES=${device} python test_evaluation.py -ep results/stnet/${exp_code}