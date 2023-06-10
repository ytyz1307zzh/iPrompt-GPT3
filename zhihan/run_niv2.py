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

TASK_NAMES = ['task066_timetravel_binary_consistency_classification', 'task070_abductivenli_incorrect_classification', 'task1573_samsum_classification', 'task065_timetravel_consistent_sentence_classification', 'task298_storycloze_correct_end_classification', 'task1728_web_nlg_data_to_text', 'task1407_dart_question_generation', 'task677_ollie_sentence_answer_generation', 'task1409_dart_text_generation', 'task1598_nyc_long_text_generation', 'task957_e2e_nlg_text_generation_generate', 'task349_squad2.0_answerable_unanswerable_question_classification', 'task226_english_language_answer_relevance_classification', 'task020_mctaco_span_based_question', 'task290_tellmewhy_question_answerability', 'task1439_doqa_cooking_isanswerable', 'task1442_doqa_movies_isanswerable', 'task242_tweetqa_classification', 'task1624_disfl_qa_question_yesno_classification', 'task520_aquamuse_answer_given_in_passage', 'task050_multirc_answerability', 'task1506_celebrity_minimal_dob_span', 'task1517_limit_classfication', 'task456_matres_intention_classification', 'task388_torque_token_classification', 'task1518_limit_answer_generation', 'task1410_dart_relationship_extraction', 'task676_ollie_relationship_answer_generation', 'task180_intervention_extraction', 'task749_glucose_reverse_cause_emotion_detection', 'task684_online_privacy_policy_text_information_type_generation', 'task958_e2e_nlg_text_generation_parse', 'task1413_dart_object_identification', 'task292_storycommonsense_character_text_generation', 'task578_curiosity_dialogs_answer_generation', 'task1597_nyc_slot_filling', 'task747_glucose_cause_emotion_detection', 'task678_ollie_actual_relationship_answer_generation', 'task1510_evalution_relation_extraction', 'task1451_drug_dose_extraction', 'task683_online_privacy_policy_text_purpose_answer_generation', 'task179_participant_extraction', 'task1411_dart_subject_identification', 'task181_outcome_extraction', 'task748_glucose_reverse_cause_event_detection', 'task621_ohsumed_yes_no_numerical_answer_generation', 'task647_answer_generation', 'task1210_atomic_classification_madeupof', 'task1215_atomic_classification_capableof', 'task1216_atomic_classification_causes', 'task1202_atomic_classification_xneed', 'task136_winowhy_knowledge_categorization', 'task1196_atomic_classification_oeffect', 'task291_semeval_2020_task4_commonsense_validation', 'task1208_atomic_classification_xreason', 'task1206_atomic_classification_isbefore', 'task1197_atomic_classification_oreact', 'task1213_atomic_classification_desires', 'task116_com2sense_commonsense_reasoning', 'task1201_atomic_classification_xintent', 'task1198_atomic_classification_owant', 'task1212_atomic_classification_hasproperty', 'task1203_atomic_classification_xreact', 'task1214_atomic_classification_xwant', 'task1200_atomic_classification_xeffect', 'task1209_atomic_classification_objectuse', 'task1204_atomic_classification_hinderedby', 'task1207_atomic_classification_atlocation', 'task1205_atomic_classification_isafter', 'task1199_atomic_classification_xattr', 'task1156_bard_analogical_reasoning_tools', 'task1159_bard_analogical_reasoning_containers', 'task1155_bard_analogical_reasoning_trash_or_treasure', 'task1157_bard_analogical_reasoning_rooms_for_containers', 'task1154_bard_analogical_reasoning_travel', 'task1158_bard_analogical_reasoning_manipulating_items', 'task1152_bard_analogical_reasoning_causation', 'task1153_bard_analogical_reasoning_affordance', 'task131_scan_long_text_generation_action_command_long', 'task129_scan_long_text_generation_action_command_short', 'task110_logic2text_sentence_generation', 'task1603_smcalflow_sentence_generation', 'task1714_convai3_sentence_generation', 'task360_spolin_yesand_response_generation', 'task574_air_dialogue_sentence_generation', 'task565_circa_answer_generation', 'task576_curiosity_dialogs_answer_generation', 'task1600_smcalflow_sentence_generation', 'task1729_personachat_generate_next', 'task1730_personachat_choose_next', 'task361_spolin_yesand_prompt_response_classification']


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
    parser.add_argument('-data_dir', type=str, default='/home/qingkaizeng/Zhihan/LLM/training_data/out-domain/niv2_english')
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

        prompt_prefix = f'Data:'
        prompt_suffix = 'Instrution:'

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
            llm_candidate_regeneration_prompt_start=prompt_prefix,
            llm_candidate_regeneration_prompt_end=prompt_suffix,
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
