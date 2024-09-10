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

