# run
start to construct your transfer set
```python
python transfer.py victim --modelname ssd300_vgg --victim_dataset nwpu-vhr --out_dir transferset/DIOR --budget 8 --queryset datasets/stealing_NWPU-VHR --batch_size 8 -d 0
```

and then start to train
```python 
 python train.py transferset/DIOR victim ssd300_vgg datasets/stealing_NWPU-VHR --budgets 8 -d 0 --batch-size 8 -e 1 --out_dir wellTrain -w 1
```
