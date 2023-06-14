# Tokyo
python train_ssl.py --dataset tokyo --w-local 0.1 --w-global 1.0 --global-batch-size 128 --global-neg-ratio 8 --grid-len 300
python train_trial.py --dataset tokyo --evaluate-every 10 --regularization 1e-5 --grid-len 300 --pretrain-path ./output/saved_model_tokyo_default_ssl

# New York
# python train_ssl.py --dataset nyc --w-local 1.0 --w-global 2.0 --global-batch-size 128 --global-neg-ratio 8 --grid-len 6
# python train_trial.py --dataset nyc --evaluate-every 10 --grid-len 6 --regularization 1e-4 --pretrain-path ./output/saved_model_nyc_default_ssl

# Chicago
# python train_ssl.py --dataset chicago --w-local 1.0 --w-global 1.0 --global-batch-size 128 --global-neg-ratio 3 --grid-len 1200
# python train_trial.py --dataset chicago --evaluate-every 10 --grid-len 1200 --regularization 1e-4  --pretrain-path ./output/saved_model_chicago_default_ssl