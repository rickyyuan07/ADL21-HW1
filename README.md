# Sample Code for Homework 1 ADL NTU 109 Spring

## Environment
```shell
# If you have conda, we recommend you to build a conda environment called "adl"
make
# otherwise
pip install -r requirements.txt
```

## Download trained models
```shell
# Please make sure you have change the argument of the downloaded model for later testing
bash download.sh
```
The downloaded model would locate at `./intent_best.pt` and `./slot_best.pt`.

## Preprocessing
```shell
# To preprocess intent detection and slot tagging datasets
bash preprocess.sh
```
Please make sure you have directories for cache, ckpt and data after preprocessing.

## Intent detection
```shell
python train_intent.py
```

## Slot tagging
```shell
python train_slot.py
```

After training, the default model would locate at `./ckpt/intent/best.pt` and `./ckpt/slot/best.pt`.
If you use the scripts to download models, move them to the corresponding directories or change the arguments in `intent_cls.sh` and `slot_tag.sh`.

## Intent Testing
```shell
bash intent_cls.sh /path/to/test.json /path/to/pred.csv
```

## Slot tagging
```shell
python test_slot.py /path/to/test.json /path/to/pred.csv
```