# morpheme level
#CUDA_VISIBLE_DEVICES=1 python main.py --mode train --config textcnn.json

# word attented on morpheme
CUDA_VISIBLE_DEVICES=1 python main2.py --mode train --config morpheme_textcnn.json
