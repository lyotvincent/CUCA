#sleep 2h && echo "ready to run evaluation"
res_path="results"
device=0


all_exp_codes=(
# "CUCAMLP_virchow2_BN_RMSE_a0.7b0.2_ep100_bs128_lr0.002_OneCycleLR"
# "CUCAMLP_virchow2_BN_RMSE_a0.4b0.5_ep100_bs128_lr0.002_OneCycleLR"
# "CUCAMLP_virchow2_BN_RMSE_a0.3b0.6_ep100_bs128_lr0.002_OneCycleLR"
# "HisToGene_resnet18_MSE_a0.6b0.3_ep100_bs1_lr0.00001_scheNoAdjust"
# "THItoGene_resnet18_MSE_a0.6b0.3_ep200_bs1_lr0.00001_scheNoAdjust"
# "MLP_virchow2_BN_RMSE_ep100_bs128_lr0.002"
"LinearProbing_virchow2_RMSE_a0.b0._ep100_bs128_lr0.002_1"
# "CUCAMLP_virchow2_BN_RMSE_a0.1b0.8_ep100_bs128_lr0.002_OneCycleLR"
# "CUCAMLP_virchow2_BN_RMSE_a0.2b0.7_ep100_bs128_lr0.002_OneCycleLR"
# "CUCAMLP_virchow2_BN_RMSE_a0.5b0.4_ep100_bs128_lr0.002_OneCycleLR"
# "CUCAMLP_virchow2_BN_RMSE_a0.6b0.3_ep100_bs128_lr0.002"
# "CUCA_virchow2_BN_RMSE_a0.2b0.7_ep100_bs128_lr0.002_ext0" # only stnet
# "CUCA_virchow2_BN_RMSE_a0.3b0.6_ep100_bs128_lr0.002_OneCycleLR"
# "hist2cell_resnet18_BN_MSE_a0.6b0.3_ep10_bs16_lr0.0001_CosineAnnealingLR"
# "ST-Net_densenet121_RMSE_a0.b0._ep50_bs32_sgdlr0.01_NoAdjust"
# "CUCAMLP_hoptimus0_BN_RMSE_a0.3b0.6_ep100_bs128_lr0.002_1"
# "CUCAMLP_gigapath_BN_RMSE_a0.3b0.6_ep100_bs128_lr0.002_1"
# "CUCAMLP_virchow2_BN_RMSE_a0.3b0.6_ep100_bs128_lr0.002_1"
# "CUCAMLP_virchow_BN_RMSE_a0.3b0.6_ep100_bs128_lr0.002_1"
# "CUCAMLP_uni_v1_BN_RMSE_a0.3b0.6_ep100_bs128_lr0.002_1"
# "CUCAMLP_conch_v1_BN_RMSE_a0.3b0.6_ep100_bs128_lr0.002_1"
# "CUCAMLP_phikon_BN_RMSE_a0.3b0.6_ep100_bs128_lr0.002_1"
# "CUCAMLP_plip_BN_RMSE_a0.3b0.6_ep100_bs128_lr0.002_1"
# "CUCA_virchow2_BN_RMSE_a0.7b0.2_ep100_bs128_lr0.002_ext0" # only her2st
# "CUCA_virchow2_BN_RMSE_a0.1b0.8_ep100_bs128_lr0.002_ext0" # only humanlung, have not finished 
)

for i in "${!all_exp_codes[@]}"; do
    exp_code="${all_exp_codes[$i]}"
    CUDA_VISIBLE_DEVICES=${device} python test_evaluation.py -ep ${res_path}/humanlung_cell2location/${exp_code}
    CUDA_VISIBLE_DEVICES=${device} python test_evaluation.py -ep ${res_path}/her2st/${exp_code}
    CUDA_VISIBLE_DEVICES=${device} python test_evaluation.py -ep ${res_path}/stnet/${exp_code}
    echo "done ${exp_code}"
done
