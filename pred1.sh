BASE_OUTPUT="predictions-0120"

for year in {2000..2002}
do
  echo "正在处理年份：$year"
  mkdir -p "$BASE_OUTPUT/$year"
  n2o-pred predict \
    --model outputs/LSTM/split_1949 \
    --dataset "input_0120/$year" \
    --output "$BASE_OUTPUT/$year" \
    --device cuda:0

  echo "年份 $year 已经处理完毕"
  echo "-----------------------"
done

echo "done!"
