export BERT_BASE_DIR=./bert/bert_default
DATASET_JSON_FILE_ROUTE="/home/lazar/Zucchabar/Datasets/QuestionClassification/dataset_mini.json"
DATASET_DIRECTORY_FOR_BERT_MODEL="data/"


# python "divide_dataset.py" $DATASET_DIRECTORY_FOR_BERT_MODEL $DATASET_JSON_FILE_ROUTE

# python "create_finetuning_data.py" \
# 	--input_data_dir="./data/" \
# 	--vocab_file="${BERT_BASE_DIR}/vocab.txt" \
# 	--meta_data_file_path="./data/meta_data" \
# 	--train_data_output_path="./data/train.tf_record" \
# 	--eval_data_output_path="./data/eval.tf_record" \




python "fit_classifier.py" \
	--mode='train_and_eval' \
	--bert_config_file=${BERT_BASE_DIR}/bert_config.json \
	--model_dir="./model/" \
	--init_checkpoint=${BERT_BASE_DIR}/bert_model.ckpt \
	--train_data_path="./data/train.tf_record" \
	--eval_data_path="./data/eval.tf_record" \
	--input_meta_data_path="./data/meta_data" \
	--strategy_type="mirror"