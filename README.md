# lumi

usage: xxx.py [-h] [-g] [-f] [-t] [-a] [-s] model_version  
 model = hhar/ motion/ uci/ shoaib
version = 10_100 / 20_120
-g : Set specific GPU
-f : Pretrain model file, default: None
-s : The name of model to be saved, default: 'model'

eg:
python pretrain.py v1 uci 20_120 -s uci : use uci dataset to train; data len = 120; final model save name = uci, config version = v1
python embedding.py v1 uci 20_120 -f uci : get represatation from a uci model; data len = 120; pretrained model name = uci

a flaw of distillation :
1.python pretrain.py v1 uci 20_120 -s uci
2.python embedding.py v1 uci 20_120 -f uci
3.python classifier.py v2 uci 20_120 -f uci -s limu_gru_v1 -l 0
4.python distill.py v1 uci 20_120 -s uci
5.python embedding_distilled.py v1 uci 20_120 -f uci -s uci
6.python classifier_distilled.py v2 uci 20_120 -f uci -s limu_gru_v1 -l 0
