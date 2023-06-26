# CluRec

### 1. Pretrain DEC

```
    python clurec/nrms_group_wa.py -d small --dec_pretrain --dec_batch 256 --dec_lr 0.0001 -c 270 --upct 10
```

### 2. Train CluRec

```
    python clurec/nrms_group_wa.py -d small -e 5 --dec_batch 256 --dec_lr 0.0001 -c 270 --upct 10
```

### 3. Train Baseline

```
    python baseline/nrms.py -d small -e 5
```


