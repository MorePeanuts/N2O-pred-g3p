BASE_OUTPUT="outputs/predictions-train-5-0510"

for num in {1000..1009}; do
  n2o-pred predict \
    --model outputs/LSTM/split_1949 \
    --dataset datasets/data_EUR_processed.pkl \
    --shap-seed $num \
    --output "$BASE_OUTPUT/$num/predictions-lstm-1949" \
    --device cuda:0 \
    --plot 44,3 56,3 56,4 91,2 93,6 &

  n2o-pred predict \
    --model outputs/LSTM/split_859 \
    --dataset datasets/data_EUR_processed.pkl \
    --shap-seed $num \
    --output "$BASE_OUTPUT/$num/predictions-lstm-859" \
    --device cuda:1 \
    --plot 44,3 56,3 56,4 91,2 93,6 &

  n2o-pred predict \
    --model outputs/LSTM/split_2869 \
    --dataset datasets/data_EUR_processed.pkl \
    --shap-seed $num \
    --output "$BASE_OUTPUT/$num/predictions-lstm-2869" \
    --device cuda:2 \
    --plot 44,3 56,3 56,4 91,2 93,6 &

  n2o-pred predict \
    --model outputs/LSTM/split_8033 \
    --dataset datasets/data_EUR_processed.pkl \
    --shap-seed $num \
    --output "$BASE_OUTPUT/$num/predictions-lstm-8033" \
    --device cuda:3 \
    --plot 44,3 56,3 56,4 91,2 93,6 &

  n2o-pred predict \
    --model outputs/LSTM/split_7702 \
    --dataset datasets/data_EUR_processed.pkl \
    --shap-seed $num \
    --output "$BASE_OUTPUT/$num/predictions-lstm-7702" \
    --device cuda:4 \
    --plot 44,3 56,3 56,4 91,2 93,6 &

  n2o-pred predict \
    --model outputs/RF/split_1949 \
    --dataset datasets/data_EUR_processed.pkl \
    --shap-seed $num \
    --output "$BASE_OUTPUT/$num/predictions-rf-1949" \
    --plot 44,3 56,3 56,4 91,2 93,6 &

  n2o-pred predict \
    --model outputs/RF/split_859 \
    --dataset datasets/data_EUR_processed.pkl \
    --shap-seed $num \
    --output "$BASE_OUTPUT/$num/predictions-rf-859" \
    --plot 44,3 56,3 56,4 91,2 93,6 &

  n2o-pred predict \
    --model outputs/RF/split_2869 \
    --dataset datasets/data_EUR_processed.pkl \
    --shap-seed $num \
    --output "$BASE_OUTPUT/$num/predictions-rf-2869" \
    --plot 44,3 56,3 56,4 91,2 93,6 &

  n2o-pred predict \
    --model outputs/RF/split_8033 \
    --dataset datasets/data_EUR_processed.pkl \
    --shap-seed $num \
    --output "$BASE_OUTPUT/$num/predictions-rf-8033" \
    --plot 44,3 56,3 56,4 91,2 93,6 &

  n2o-pred predict \
    --model outputs/RF/split_7702 \
    --dataset datasets/data_EUR_processed.pkl \
    --shap-seed $num \
    --output "$BASE_OUTPUT/$num/predictions-rf-7702" \
    --plot 44,3 56,3 56,4 91,2 93,6 &

done

wait

echo "All predictions completed."
