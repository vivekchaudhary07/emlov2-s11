# chmod +x scripts/train_multinode.sh
# scripts/train_multinode.sh 1.1.1.19 0
master_node_ip=$1
node_rank=$2

MASTER_PORT=29500 MASTER_ADDR=master_node_ip WORLD_SIZE=2 NODE_RANK=node_rank python src/train.py experiment=cifar.yaml
