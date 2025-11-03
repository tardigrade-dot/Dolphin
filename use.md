/Users/larrynote/.cache/huggingface/hub/models--ByteDance--Dolphin

/Users/larrynote/coderesp/dots.ocr/weights/Dolphin

python demo_page_hf.py --model_path ByteDance/Dolphin --input_path ./demo/page_imgs/page_1.jpeg --save_dir ./results

python demo_page_hf.py --model_path ByteDance/Dolphin --input_path ./demo/page_imgs/page_2.jpeg --save_dir ./results

daodufuke.pdf

python demo_page_hf.py --model_path ByteDance/Dolphin --input_path ./demo/page_imgs/daodufuke.pdf --save_dir ./results_daodufuke --max_batch_size 10

/Users/larrynote/coderesp/Dolphin/demo/page_imgs/daodufuke.pdf

/Users/larrynote/datalib/检讨.pdf

python demo_page_hf.py --model_path ByteDance/Dolphin --input_path /Users/larrynote/datalib/jiantao.pdf --save_dir ./results_jiantao --max_batch_size 1


python demo_page_hf.py --model_path ByteDance/Dolphin --input_path '/Users/larrynote/ai/mnist/ocr/pdfs/我所体验的社会主义 (菊地昌典) (Z-Library).pdf' --save_dir ./results_jiti --max_batch_size 1


python demo_page_hf.py --model_path ByteDance/Dolphin --input_path '/Users/larrynote/ai/mnist/ocr/pdfs/改造：1949-1957年的知识分子 (于风政) (Z-Library).pdf' --save_dir ./results_gaizao --max_batch_size 2

# 处理单个文档图像

python demo_page_hf.py --model_path ByteDance/Dolphin --input_path ./results_gaizao/page_239.png --save_dir ./results_gaizao/page_239


python demo_page_hf.py --model_path /Users/larry/coderesp/Dolphin/pretrained_models/Dolphin --input_path ./results_gaizao/page_239.png --save_dir ./results_gaizao/page_239


# 使用
python demo_page_hf.py --model_path ByteDance/Dolphin --input_path '/Users/larry/Documents/权力学 The Technology of Power (阿夫托尔哈诺夫（Abdurakhman Avtorkhanov）) (Z-Library).pdf' --save_dir ./results_power --max_batch_size 1


python demo_page_hf.py --model_path ByteDance/Dolphin --input_path '/Users/larry/Documents/蜉蝣天地话沧桑——九十自述 (资中筠) (Z-Library).pdf' --save_dir ./results_fuyou --max_batch_size 1

/Users/larry/miniconda3/envs/ocr311/bin/python demo_page_hf.py --model_path "/Volumes/sw/pretrained_models/Dolphin-1.5" --input_path '/Users/larry/Documents/蜉蝣天地话沧桑——九十自述 (资中筠) (Z-Library).pdf' --save_dir /Volumes/sw/ocr_result/results_fuyou --max_batch_size 1

/Users/larry/miniconda3/envs/dolphin311/bin/python demo_page.py  --model_path "/Volumes/sw/pretrained_models/Dolphin-1.5" --input_path '/Volumes/sw/MyDrive/data_src/独立宣言 一种全球史 (（美）大卫·阿米蒂奇) (Z-Library).pdf' --save_dir /Volumes/sw/ocr_result/results_duli --max_batch_size 2

/Users/larry/miniconda3/envs/dolphin311/bin/python demo_page.py  --model_path "/Volumes/sw/pretrained_models/Dolphin-1.5" --input_path '/Volumes/sw/MyDrive/data_src/独立宣言 一种全球史 (（美）大卫·阿米蒂奇) (Z-Library).pdf' --save_dir /Volumes/sw/ocr_result/results_duli --max_batch_size 2


/Users/larry/miniconda3/envs/dolphin311/bin/dolphin-run --model_path "/Volumes/sw/pretrained_models/Dolphin-1.5" --input_path '/Volumes/sw/MyDrive/data_src/独立宣言 一种全球史 (（美）大卫·阿米蒂奇) (Z-Library).pdf' --save_dir /Volumes/sw/ocr_result/results_duli --max_batch_size 2


/Users/larry/miniconda3/envs/dolphin311/bin/dolphin-run --model_path "/Volumes/sw/pretrained_models/Dolphin-1.5" --input_path '/Volumes/sw/MyDrive/data_src/果壳阅读 错觉在或不在，时间都在 (（英）克劳迪娅·哈蒙德著；桂江城译 etc.) (Z-Library).pdf' --save_dir /Volumes/sw/ocr_result/results_duli --max_batch_size 2