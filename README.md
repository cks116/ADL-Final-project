# ADL Final project

## Train model

training tag model

```bash
python3 train.py --tag
```

training predict model

```bash
python3 train.py --predict
```

If you want to use train + dev set to train model.

```bash
python3 train.py --largeset --[tag|predict]
```

## Testing

If you don't train model, have to download model first.

```bash
bash download.sh
```

Then, you can test

```bash
python3 test.py <path_to_test_set_directory>
```
