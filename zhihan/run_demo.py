from imodelsx import explain_dataset_iprompt, get_add_two_numbers_dataset
import openai
import sys
openai.api_key = sys.argv[1]

# get a simple dataset of adding two numbers
data = [
    {"id": "task1728-00e9ac342c3c45c0858d1275199f491a", "instruction": "You will be given one or more triples. The second part of each triple shows the relation between the first and the third element. Your task is to write a simple and short piece of text (sentence(s)) that describes the triples in natural language.", "input": "Acta_Palaeontologica_Polonica | LCCN_number | 60040714\nActa_Palaeontologica_Polonica | abbreviation | \"Acta Palaeontol. Pol.\"\nActa_Palaeontologica_Polonica | academicDiscipline | Paleobiology", "target": "The Acta Palaeontologica Polonica has a LCCN number of 60040714 and the abbreviation of Acta Palaeontol. Pol. It comes under the academic discipline of Paleobiology."},
    {"id": "task1728-346eab2a5c8642868ca23a3798cf48c5", "instruction": "You will be given one or more triples. The second part of each triple shows the relation between the first and the third element. Your task is to write a simple and short piece of text (sentence(s)) that describes the triples in natural language.", "input": "Al_Asad_Airbase | operatingOrganisation | United_States_Air_Force\nAl_Asad_Airbase | location | Iraq\nAl_Asad_Airbase | runwayLength | 3990.0\nAl_Asad_Airbase | runwayName | \"09L/27R\"", "target": "Operated by the United States Air Force, Al Asad Airbase is in Iraq. Its runway name is 09L/27R and its runway length is 3990.0."},
    {"id": "task1728-4dcdbbab70484034b64fd65bdf0d4d92", "instruction": "You will be given one or more triples. The second part of each triple shows the relation between the first and the third element. Your task is to write a simple and short piece of text (sentence(s)) that describes the triples in natural language.", "input": "Bionico | country | Mexico\nMexico | language | Mexican_Spanish", "target": "Bionico is a food found in Mexico where Mexican Spanish is spoken."}
]

input_strings = [x['input'] for x in data]
output_strings = [x['target'] for x in data]

# explain the relationship between the inputs and outputs
# with a natural-language prompt string
prompts, metadata = explain_dataset_iprompt(
    input_strings=input_strings,
    output_strings=output_strings,
    checkpoint='facebook/opt-125m', # which language model to use
    num_learned_tokens=64, # how long of a prompt to learn
    n_shots=3, # shots per example
    n_epochs=1, # how many epochs to search
    verbose=1, # how much to print
    batch_size=3,
# generation_temp=1.0,
# generation_top_p=0.75,
    max_n_datapoints=5,
    max_n_steps=10000,
    openai_model_name='text-davinci-003',
    llm_candidate_regeneration_prompt_start='Data:',
    llm_candidate_regeneration_prompt_end='Instruction:',
    # llm_float16=True, # whether to load the model in float_16
)
# --------
# prompts is a list of found natural-language prompt strings