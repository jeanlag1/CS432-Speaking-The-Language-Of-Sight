CUDA_VISIBLE_DEVICES=0,1,2,3 \
torchrun \
--nproc_per_node=1 \
--master_port 30000 \
train_retrieval.py \
--config configs/retrieval_coco.yaml \
--output_dir output/retrieval_coco