# 消融实验命令（chest_3views）
# SPS init 文件: data/369-sps/init_chest_50_3views.npy
# Random init 文件: data/369/init_chest_50_3views.npy
# GAR flags: --enable_fsgs_proximity --gar_proximity_threshold 0.05 --gar_proximity_k 5 --no_gar_adaptive_threshold --no_gar_progressive_decay --gar_new_per_source 1 --gar_max_candidates 2000
# ADM flags: --enable_kplanes --adm_resolution 64 --adm_feature_dim 32 --adm_decoder_hidden 128 --adm_decoder_layers 3 --kplanes_lr_init 0.005 --lambda_plane_tv 0.0005 --adm_warmup_iters 15000 --adm_max_range 0.3 --adm_view_adaptive --adm_zero_mean --adm_zero_mean_mode density_confidence

# ├── SPS only
# ├── GAR only  
# ├── ADM only
# ├── SPS + GAR
# ├── SPS + ADM
# ├── GAR + ADM
# └── Full SPAGS (already done: 27.11)
