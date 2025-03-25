architecture=CUCA
backbone=virchow2
loss_main=RMSE


lr_rate=0.002
max_epochs=100
proj_dim=512

pre_extracted=0

device=0


lambda_main=0.3
lambda_rec=0.6
exp_code=${architecture}_${backbone}_BN_${loss_main}_a${lambda_main}b${lambda_rec}_ep${max_epochs}_d${proj_dim}_lr${lr_rate}_ext${pre_extracted}

## training the model on the humanlung dataset
CUDA_VISIBLE_DEVICES=${device} python main.py -c cfgs/cfgs_lung.yaml \
--opts CKPTS exp_code ${exp_code} HyperParams pre_extracted ${pre_extracted} HyperParams max_epochs ${max_epochs} HyperParams proj_dim ${proj_dim} HyperParams loss_main ${loss_main} HyperParams lambda_main ${lambda_main} HyperParams lambda_rec ${lambda_rec} HyperParams architecture ${architecture} HyperParams backbone ${backbone} HyperParams lr_rate ${lr_rate}
## evaluating the model on the humanlung dataset
CUDA_VISIBLE_DEVICES=${device} python test_evaluation.py -ep results/humanlung_cell2location/${exp_code}


## training the model on the her2st dataset
CUDA_VISIBLE_DEVICES=${device} python main.py -c cfgs/cfgs_her2st.yaml \
--opts CKPTS exp_code ${exp_code} HyperParams pre_extracted ${pre_extracted} HyperParams max_epochs ${max_epochs} HyperParams proj_dim ${proj_dim} HyperParams loss_main ${loss_main} HyperParams lambda_main ${lambda_main} HyperParams lambda_rec ${lambda_rec} HyperParams architecture ${architecture} HyperParams backbone ${backbone} HyperParams lr_rate ${lr_rate}
## evaluating the model on the her2st dataset
CUDA_VISIBLE_DEVICES=${device} python test_evaluation.py -ep results/her2st/${exp_code}


lambda_main=0.2
lambda_rec=0.7
exp_code=${architecture}_${backbone}_BN_${loss_main}_a${lambda_main}b${lambda_rec}_ep${max_epochs}_d${proj_dim}_lr${lr_rate}_ext${pre_extracted}

## training the model on the stnet dataset
CUDA_VISIBLE_DEVICES=${device} python main.py -c cfgs/cfgs_stnet.yaml \
--opts CKPTS exp_code ${exp_code} HyperParams pre_extracted ${pre_extracted} HyperParams max_epochs ${max_epochs} HyperParams proj_dim ${proj_dim} HyperParams loss_main ${loss_main} HyperParams lambda_main ${lambda_main} HyperParams lambda_rec ${lambda_rec} HyperParams architecture ${architecture} HyperParams backbone ${backbone} HyperParams lr_rate ${lr_rate}
## evaluating the model on the stnet dataset
CUDA_VISIBLE_DEVICES=${device} python test_evaluation.py -ep results/stnet/${exp_code}