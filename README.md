# SmoothCAM

# environment setup
```
conda create -n env_name python=3.9
conda activate env_name
cd ./AGCAM
pip install -r requirements.txt
```

###  the weakly-supervised localization test
```
python localization.py --method=smoothcam --data_root=./ILSVRC --threshold=0.5
```


### the pixel perturbation test
```
python save_h5.py --method=agcam  --save_root=.\saveroot --data_root=.\ILSVRC
python ABPC.py --method=agcam --h5_root=.\saveroot --csv=True --file=True
```

Please download the model from the ‘[jx_vit_base_p16_224-80ecf9dd](https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_p16_224-80ecf9dd.pth)’ link first.
