import os
import sys
import json
import random
import argparse
import numpy as np
from typing import Dict, List
sys.path.append('.')
from imodelsx.iprompt.api import explain_dataset_iprompt
import openai

TASK_NAMES = ['boolean_expressions', 'causal_judgement', 'date_understanding', 'disambiguation_qa', 'dyck_languages', 'formal_fallacies', 'geometric_shapes', 'hyperbaton', 'logical_deduction_five_objects', 'logical_deduction_seven_objects', 'logical_deduction_three_objects', 'movie_recommendation', 'multistep_arithmetic_two', 'navigate', 'object_counting', 'penguins_in_a_table', 'reasoning_about_colored_objects', 'ruin_names', 'salient_translation_error_detection', 'snarks', 'sports_understanding', 'temporal_sequences', 'tracking_shuffled_objects_five_objects', 'tracking_shuffled_objects_seven_objects', 'tracking_shuffled_objects_three_objects', 'web_of_lies', 'word_sorting']


def read_jsonl_as_list(path: str):
    assert path.endswith('.jsonl')
    with open(path, 'r', encoding='utf8') as fin:
        result = []
        for line in fin:
            data = json.loads(line.strip())
            result.append(data)
    # print(f'Read {len(result)} data from {path}')
    return result


def save_list_as_jsonl(path: str, data):
    assert path.endswith('.jsonl')
    with open(path, 'w', encoding='utf8') as fout:
        for instance in data:
            fout.write(json.dumps(instance))
            fout.write('\n')
    # print(f'Saved {len(data)} data to {path}')


def mean(array):
    assert isinstance(array, list)
    return sum(array) / len(array)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-data_dir', type=str, default='/home/qingkaizeng/Zhihan/LLM/training_data/out-domain/bbh/direct_answer')
    parser.add_argument('-demo_file', type=str, default='demo.jsonl')
    parser.add_argument('-output_file', type=str, default='iprompt_generated_instruction.json')
    parser.add_argument('-api_key', type=str, required=True)
    args = parser.parse_args()

    openai.api_key = args.api_key

    for i, task in enumerate(TASK_NAMES):

        print(f"{'-' * 15}{task}({i + 1}/{len(TASK_NAMES)}){'-' * 15}")
        demo_file = os.path.join(args.data_dir, task, args.demo_file)
        demo_list = read_jsonl_as_list(demo_file)
        output_file = os.path.join(args.data_dir, task, args.output_file)

        if os.path.exists(output_file):
            print(f'{task}: {args.output_file} exists, skip ...')
            continue

        input_strings = [x['input'] for x in demo_list]
        output_strings = [x['target'] for x in demo_list]

        # explain the relationship between the inputs and outputs
        # with a natural-language prompt string
        prompts, metadata = explain_dataset_iprompt(
            input_strings=input_strings,
            output_strings=output_strings,
            checkpoint='facebook/opt-125m',  # which language model to use
            num_learned_tokens=300,  # how long of a prompt to learn
            n_shots=3,  # shots per example
            n_epochs=1,  # how many epochs to search
            verbose=0,  # how much to print
            batch_size=3,
            # generation_temp=1.0,
            # generation_top_p=0.75,
            max_n_datapoints=300,
            max_n_steps=100,
            openai_model_name='text-davinci-003',
            llm_candidate_regeneration_prompt_start='Data:',
            llm_candidate_regeneration_prompt_end='Instruction:',
            loss_wait_patience=10
        )

        loss = metadata['prefix_train_loss']
        json.dump(
            {'prompt': prompts, 'loss': loss},
            open(output_file, 'w', encoding='utf8'),
            indent=4, ensure_ascii=False
        )


if __name__ == '__main__':
    main()
