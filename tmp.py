MASTER_PORT=29500 MASTER_ADDR=65.0.123.80 WORLD_SIZE=2 NODE_RANK=0 python src/train.py experiment=cifar.yaml
MASTER_PORT=29500 MASTER_ADDR=65.0.123.80 WORLD_SIZE=2 NODE_RANK=1 python src/train.py experiment=cifar.yaml

scp -i .\key_new.pem -r ubuntu@65.0.123.80:
scp -i .\key_new.pem -r ubuntu@52.66.223.228:
