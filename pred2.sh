BASE_OUTPUT="predictions-model-859-2"

for year in {2000..2020}; do
  echo "正在处理年份：$year"
  mkdir -p "$BASE_OUTPUT/$year"
  n2o-pred predict \
    --model outputs/LSTM/split_859 \
    --dataset "input_0223/$year" \
    --output "$BASE_OUTPUT/$year" \
    --device cuda:2

  echo "年份 $year 已经处理完毕"
  echo "-----------------------"
done

echo "done!"
