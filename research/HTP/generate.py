import sys
import os
os.environ["TORCH_DISTRIBUTED_DEBUG"] = "INFO" 
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
import paddle
import pickle
import logging
import numpy as np
from paddlenlp import transformers
from paddle.io import Dataset
from paddlenlp.data import DataCollatorForSeq2Seq
from paddlenlp.trainer import PdArgumentParser, get_last_checkpoint,Seq2SeqTrainingArguments
from paddlenlp.transformers import AutoTokenizer, AutoModelForCausalLM
from bloomModel.HTP_Model import BloomForCausalLM
from paddlenlp.datasets import InTokensIterableDataset, InTokensMapDataset, load_dataset
from paddlenlp.transformers.attention_utils import MultiHeadAttention
from paddlenlp.transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    AutoConfig,
)
from argument import (
    DataArgument,
    GenerateArgument,
    ModelArgument,
    QuantArgument,
    TrainingArguments,

)
from trainer_seq2seq import Seq2SeqTrainer
from peft.prompt_tuning import *
from peft.prompt_config import *
logger = logging.getLogger(__name__)



def expend(tokens, number, tag_token):
    return tokens[:number] + [tag_token] * max(0, number - len(tokens))

def table_encode(t, tokenizer):
    flat_list = [item for sublist in t for item in sublist]
    tokenized = tokenizer(flat_list, add_special_tokens=False)['input_ids']

    it = iter(tokenized)
    return [[next(it) for _ in sublist] for sublist in t]


def merge_table(table_list, tag_token):
    H, VAL = table_list
    result = [[0 for _ in range(len(H[0]))] for _ in range(len(H))]
    for i in range(len(H)):
        for j in range(len(H[0])):
            result[i][j] = expend(H[i][j], MAX_H, tag_token) + expend(VAL[i][j], MAX_VAL, tag_token) 
    return np.array(result)

def table_crop_pad(h, t, max_row, max_col, pad_str):
 
    new_table = []
    rows = []
    for x in t:
        if len(x)!=0:
            new_table.append(x)
            rows.append(len(x))
    min_row = min(rows)

    
    t = [[row[i] for row in new_table ] for i in range(min_row)]
    

    t_array = np.array(t,dtype='<U64')
    n, m = t_array.shape
   
    h_array = np.array(h,dtype='<U64')
    h_array = h_array.reshape((1, h_array.shape[0]))
    h_array = np.repeat(h_array, n, axis=0)

    rows_to_crop = max(0, n - max_row)
    cols_to_crop = max(0, m - max_col)
    cropped_t_array = t_array[:max_row, :max_col]
    cropped_h_array = h_array[:max_row, :max_col]
    padded_t_array = np.pad(cropped_t_array, ((0, max_row - cropped_t_array.shape[0]), (0, max_col - cropped_t_array.shape[1])), mode='constant', constant_values=pad_str)
    padded_h_array = np.pad(cropped_h_array, ((0, max_row - cropped_h_array.shape[0]), (0, max_col - cropped_h_array.shape[1])), mode='constant', constant_values=pad_str)


    padded_t_list = padded_t_array.tolist()
    padded_h_list = padded_h_array.tolist()
    
    return padded_h_list, padded_t_list

def process_tables(tokenizer, value, head):
    pad_str = '<pad>'
    tag_token = 3

    head, value = table_crop_pad(head, value, max_row, max_col, pad_str)

    head = table_encode(head, tokenizer)
    value = table_encode(value, tokenizer)

    semantic_table = merge_table([head, value], tag_token)

    return semantic_table

def convert_to_half(model):
    for layer in model.sublayers():
        if isinstance(layer, nn.Linear):
            layer.weight.set_value(paddle.cast(layer.weight, 'float16'))
            if layer.bias is not None:
                layer.bias.set_value(paddle.cast(layer.bias, 'float16'))







def main():
    parser = PdArgumentParser((ModelArgument, DataArgument, Seq2SeqTrainingArguments))
    if len(sys.argv) >= 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file_and_cmd_lines()
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()


    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    

    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    hard_codes = paddle.load(data_args.hard_code_file)
    index_labels = paddle.load(data_args.cluster_index_file)

    config = AutoConfig.from_pretrained(model_args.model_name_or_path, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)
    model =  BloomForCausalLM.from_pretrained(model_args.model_name_or_path, config = config, dtype="float16")

    global max_row
    global max_col 
    max_row = config.n_ttx_row
    max_col = config.n_ttx_col

    prefix_config = PromptTuningConfig(
    num_prefix_tokens = config.instance_prompt_length,
    hidden_size = config.hidden_size
    )


    model = PromptTuningModelForCausalLM(   
                model=model,
                pt_config=prefix_config,
            )
    print(model)
   
    for name, weight in model.state_dict().items():
        weight.stop_gradient = True

    model_state_dict = paddle.load(model_args.model_name_or_path + '/model_state.pdparams', return_numpy=True)
    model.set_state_dict(model_state_dict)

    convert_to_half(model)

    model.print_trainable_parameters()
                


    class dataset(Dataset):
        
        def __init__(self, test_file_path, max_source_length,max_target_length, columns,tokenizer,prefix,ignore_pad_token_for_loss):
            
            self.test_file_path = test_file_path 
            self.test_data = self.get_test_file()
            self.max_source_length =max_source_length
            self.max_target_length =max_target_length
            self.max_seq_length =  max_source_length + max_target_length
            self.tokenizer = tokenizer
            self.columns = columns
            self.prefix = prefix
            self.ignore_pad_token_for_loss = ignore_pad_token_for_loss
            
            
        def get_test_file(self):
            data = paddle.load(self.test_file_path)
            return data
        
        def __getitem__(self, idx):
           
            model_inputs = {}
            
            item = self.test_data[idx]
            table_column, text_column, target_column = self.columns
            table, text, target = item[table_column], item[text_column], item[target_column]
            head, value = table['head'], table['value']
            semantic_table_ids = process_tables(self.tokenizer, value, head)
            
            
            text = str(text)

            prompt = self.prefix + text
            a_ids = self.tokenizer.encode(text=prompt, add_special_tokens=False)['input_ids']
            b_ids = self.tokenizer.encode(text=str(target), add_special_tokens=False)['input_ids']
            tag_ids = self.tokenizer.encode(text="这个表格数据所生成的图表代码是:", add_special_tokens=False)['input_ids']

            
            if len(a_ids) > self.max_source_length-1:
                a_ids = a_ids[: self.max_source_length-1 ]


            a_ids = [self.tokenizer.bos_token_id] + a_ids 
            left_pad_len = self.max_source_length-len(a_ids)
            source_ids = [self.tokenizer.pad_token_id]*left_pad_len + a_ids
           
            input_ids = source_ids + tag_ids 

            labels = b_ids


            model_inputs["input_ids"]= input_ids
            model_inputs["labels"]=labels
            model_inputs["semantic_table_ids"] = semantic_table_ids
            model_inputs["hard_codes"] = hard_codes
            model_inputs["index"] = index_labels[idx]
            
           
            return model_inputs
        
       

        def __len__(self):
            return (len(self.test_data))

    def print_dataset_example(example):
        print("input_ids",example["input_ids"])
        print("inputs", tokenizer.decode(example["input_ids"]))
        print("label_ids", example["labels"])
        print("labels", tokenizer.decode(example["labels"]))

    
    global MAX_H
    global MAX_VAL
    MAX_H=data_args.MAX_H
    MAX_VAL=data_args.MAX_VAL

    table_column = data_args.table_column
    text_column = data_args.text_column
    target_column = data_args.target_column

    if training_args.do_predict:
        predict_dataset = dataset(data_args.test_file, config.max_source_length,data_args.max_target_length, (table_column, text_column, target_column),tokenizer,'',True)
        print_dataset_example(predict_dataset[0])
    

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=DataCollatorForSeq2Seq(
                tokenizer=tokenizer,
                max_length=data_args.max_length,
                padding=False,
                return_tensors="np",
            ),
        compute_metrics= None,
        )

    
    results = {}
   
    if training_args.do_predict:
        predict_results = trainer.predict(predict_dataset, metric_key_prefix="predict", max_length=data_args.max_target_length, do_sample=True, top_p=0.75, temperature=0.1)
    
        metrics = predict_results.metrics
        trainer.log_metrics("predict", metrics)
        trainer.save_metrics("predict", metrics)

        if trainer.is_world_process_zero():
        
            predictions = tokenizer.batch_decode(
                predict_results.predictions, skip_special_tokens=True, clean_up_tokenization_spaces=True
            )
            predictions = [pred.strip() for pred in predictions]
            labels = tokenizer.batch_decode(
                predict_results.label_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
            )
            labels = [label.strip() for label in labels]
            output_prediction_file = os.path.join(training_args.output_dir, "generated_predictions.txt")
            with open(output_prediction_file, "w", encoding="utf-8") as writer:
                for p, l in zip(predictions, labels):
                    res = json.dumps({"labels": l, "predict": p}, ensure_ascii=False)
                    writer.write(f"{res}\n")
    return results



if __name__ == "__main__":
    main()
