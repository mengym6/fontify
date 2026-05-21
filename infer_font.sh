CUDA_VISIBLE_DEVICES=0,1 python eval/inference_font.py \
--out_dir result/UC64 \
--batch_size 2 \
--ckpt_path ckpt/checkpoint-1.pth \
--prompt_json eval/prompts.json \
--num_gpus 2 \
--ref_dir fontdata_example/font/train/chinese \
--source_dir  fontdata_example/font/train/source/ \
--gen_dir fontdata_example/font/test_unknown_content/chinese \
