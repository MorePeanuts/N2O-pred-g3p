n2o-pred predict \
  --model outputs/LSTM/split_1949 \
  --dataset datasets/data_EUR_processed.pkl \
  --output predictions-model-1949 \
  --device cuda:0

n2o-pred predict \
  --model outputs/LSTM/split_859 \
  --dataset datasets/data_EUR_processed.pkl \
  --output predictions-model-859 \
  --device cuda:0

n2o-pred predict \
  --model outputs/LSTM/split_2869 \
  --dataset datasets/data_EUR_processed.pkl \
  --output predictions-model-2869 \
  --device cuda:0

n2o-pred predict \
  --model outputs/LSTM/split_8033 \
  --dataset datasets/data_EUR_processed.pkl \
  --output predictions-model-8033 \
  --device cuda:0

n2o-pred predict \
  --model outputs/LSTM/split_8294 \
  --dataset datasets/data_EUR_processed.pkl \
  --output predictions-model-8294 \
  --device cuda:0
