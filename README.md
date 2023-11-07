# Dyck language experiments

Debug config:
```
python main.py \
  --output_dir=/net/nfs/allennlp/willm/dyck-transformers/test \
  --n_train=1000 \
  --n_eval=100 \
  --eval_threshold=10 \
  --save_threshold=1000
```

Actual run config:
```
python main.py \
  --output_dir=/net/nfs/allennlp/willm/dyck-transformers/test \
  --n_train=1000000 \
  --n_eval=5000 \
  --eval_threshold=1000 \
  --save_threshold=5000
```