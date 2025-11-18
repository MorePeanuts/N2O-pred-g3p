echo "Testing train function"
uv run n2o-pred train --model rf --max-split 3 --split-seed 123 --n-estimators 10 --output outputs/test_rf
uv run n2o-pred train --model rnn-obs --max-split 3 --split-seed 123 --max-epochs 5 --patience 3 --output outputs/test_rnn_obs
uv run n2o-pred train --model rnn-daily --max-split 3 --split-seed 123 --max-epochs 5 --patience 3 --output outputs/test_rnn_daily

echo "Testing compare function"
uv run n2o-pred compare --models outputs/test_rf outputs/test_rnn_obs outputs/test_rnn_daily --output outputs/test_compare

echo "Testing predict function"
uv run n2o-pred predict --model outputs/test_rf/split_1346 --dataset datasets/data_EUR_processed.pkl
