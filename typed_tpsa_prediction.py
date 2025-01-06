import os
import pandas as pd
import dspy
from dotenv import load_dotenv
import matplotlib.pyplot as plt
from openai import OpenAI
from dspy.teleprompt import MIPROv2
import numpy as np
from sklearn.metrics import mean_squared_error
from pydantic import BaseModel
import deepchem as dc
from loguru import logger
from typing import List
from tpsa_random_pubchem import describe_molecule, load_smarts_patterns


load_dotenv((os.path.join('.env')))
save_path = os.path.join('tpsa_saved_data')
api_key = os.getenv('OPEN_API_KEY')
turbo = dspy.LM(model='gpt-4o-mini', max_tokens=7000, api_key=api_key)#gpt-4o-2024-08-06


class TPSAResponse(BaseModel):
    contributor_description: str
    tpsa_numbers_list: List[float]


def predict_tpsa_straight(molecule):
    prompt = (f"Predict the numerical value of the topological surface area, TPSA, for a molecule "
              f"described by the SMILES code, {molecule}. ")

    client = OpenAI(
        api_key=api_key,
    )
    response = client.beta.chat.completions.parse(
        messages=[
            {
                "role": "user",
                "content": f"{prompt}",
            }
        ],
        model='gpt-4o-mini',#gpt-4o-2024-08-06
        response_format=TPSAResponse,
    )
    tpsa_numbers_list = response.choices[0].message.parsed.tpsa_numbers_list
    tpsa_numbers_list = sum(tpsa_numbers_list)
    return tpsa_numbers_list


# Initialize tracking values if needed
tracking_tpsa_values = []
successful_file = 'successful_examples.csv'
unsuccessful_file = 'unsuccessful_examples.csv'

# Create the CSV files if they don't exist
if not os.path.isfile(os.path.join(save_path, successful_file)):
    pd.DataFrame(columns=["Question", "Answer"]).to_csv(os.path.join(save_path, successful_file), index=False)
if not os.path.isfile(os.path.join(save_path, unsuccessful_file)):
    pd.DataFrame(columns=["Question", "Answer"]).to_csv(os.path.join(save_path, unsuccessful_file), index=False)


def count_no_characters(input_string):
    n_count = input_string.count('N') + input_string.count('n')
    o_count = input_string.count('O') + input_string.count('o')

    return n_count, o_count


def tpsa_match_metric(answer, pred, trace=None):
    # Extract the true value and prediction
    answer_value = float(answer.answer)
    pred_value = pred[0].answer.tpsa_numbers_list

    # Convert the values to a common format for saving
    question_str = f"True TPSA: {answer_value}"
    answer_str = f"Predicted TPSA: {pred_value}"

    # Create a dictionary entry for the current example
    example_entry = {"Question": question_str, "Answer": answer_str, "SMILES": answer.question}

    if trace is None:  # during optimization
        logger.info(f'optimization: true value, ({answer_value}, prediction, {pred_value}))')
        return -abs(answer_value - pred_value)
    else:  # for bootstrapping
        if abs(answer_value - pred_value) < 20:
            logger.info(f'Keeping for bootstrap. true, ({answer_value}, prediction, {pred_value}))')

            # Check if the predicted value is unique enough for bootstrapping
            if round(pred_value, 1) not in tracking_tpsa_values:  # Ensure variety in batches
                tracking_tpsa_values.append(round(pred_value, 1))
                if len(tracking_tpsa_values) > 5:  # Keep track of only the most recent values
                    del tracking_tpsa_values[0]

                # Log the successful example to CSV
                successful_df = pd.read_csv(os.path.join(save_path, successful_file))
                successful_df = successful_df._append(example_entry, ignore_index=True)
                successful_df.to_csv(os.path.join(save_path, successful_file), index=False)

                return True
            else:
                return False
        else:
            logger.info(f'Dropping from bootstrap. true, ({answer_value}, prediction, {pred_value}))')

            # Log the unsuccessful example to CSV
            unsuccessful_df = pd.read_csv(os.path.join(save_path, unsuccessful_file))
            unsuccessful_df = unsuccessful_df._append(example_entry, ignore_index=True)
            unsuccessful_df.to_csv(os.path.join(save_path, unsuccessful_file), index=False)

            return False


class MiproDescription(dspy.Signature):
    question = dspy.InputField(desc="smiles code for a molecule to be analyzed for predicting TPSA ")
    answer: TPSAResponse = dspy.OutputField(desc="A text summary for how each identified group effects the TPSA "
                                                 "value that MUST be followed by a list of numerical values (floats). "
                                                 "Even if there is only a single value to return it must be a list. ")


class MiproTPSA(dspy.Module):
    def __init__(self):
        super().__init__()
        predictor = dspy.TypedPredictor(MiproDescription)# wrap_json=True
        self.prog = predictor


    def forward(self, question, answer):
        group_descriptions = describe_molecule(question, functional_groups)
        if group_descriptions:
            group_descriptions = group_descriptions['Assigned Functional Groups']
        logger.info(f'smiles and group description, {question}, {group_descriptions}')
        if group_descriptions:
            data_table = []
            for group in descriptive_tpsa_data['group']:
                if group in group_descriptions:
                    data_table.append(descriptive_tpsa_data[descriptive_tpsa_data['group'] == group].to_string(index=False))

        else:
            data_table = 'No data required. '
        num_n, num_o = count_no_characters(question)
        total_hits = sum([num_o, num_n])
        prompt_a = f"Help predict the numerical value of the topological surface area, TPSA, for a molecule described by the SMILES code, {question}, "
        prompt_b = f"which can be described as comprising: {group_descriptions}. "
        prompt_c = f"Return a text summary in the field contributor_description and ALWAYS end with a LIST of numbers in the variable tpsa_numbers_list that shows how each group contributes to the total TPSA value for the molecule. tpsa_numbers_list must be a list even for a single value. "
        prompt_d = f"provide one value from the table for each of the {num_n} nitrogen atoms, and {num_o} oxygen atoms. "
        prompt_e = f"Determine the contributions of each of those {total_hits} groups to the TPSA value using this table: {data_table} . There may be multiple occurrences of some groups which should be treated additively. Groups containing just carbon, either aliphatic or aromatic, do not increase the TPSA value. "
        prompt_f = "Respond with a single JSON object. You MUST use this format: "
        prompt_combined = prompt_a + prompt_b + prompt_c + prompt_d + prompt_e + prompt_f
        tpsa_parts = self.prog(question=prompt_combined)
        tpsa_parts.answer.tpsa_numbers_list = sum(tpsa_parts.answer.tpsa_numbers_list)
        return tpsa_parts, group_descriptions, num_n, num_o, data_table


def load_pubchem_data(file_path):
    """Load PubChem data from a csv file."""
    return pd.read_csv(file_path)


def create_examples(data):
    """Create examples from the given data."""
    return [dspy.Example(augmented=True, question=row['CanonicalSMILES'], answer=str(row['TPSA'])).with_inputs("question", "answer") for
            index, row in data.iterrows()]


def create_plot(x, y, marker, file_name, predictions, path, model_name):
    plt.figure(figsize=(10, 5))
    plt.scatter(predictions[x], predictions[y], label='LLM Prediction', marker=marker)

    # Calculate RMSE for the selected model
    rmse = np.sqrt(mean_squared_error(predictions[x], predictions[y]))

    # Line of perfect prediction
    plt.plot([15, 75], [15, 75], 'r--', label='Perfect Prediction')

    # Display RMSE on the plot
    if model_name == 'tpsa_model':
        model_name = 'tpsa_model_full'
    plt.text(0.05, 0.95, f'{model_name} Prediction RMSE: {rmse:.2f}', transform=plt.gca().transAxes,
             fontsize=16, verticalalignment='top', bbox=dict(facecolor='white', alpha=0.5))

    plt.ylabel('LLM Predicted TPSA', fontsize=16)
    plt.xlabel('Calculated Pubchem TPSA', fontsize=16)
    plt.title(f'LLM Predicted TPSA vs Calculated TPSA', fontsize=16)
    plt.legend(loc='lower right')

    plt.tight_layout()
    plt.savefig(os.path.join(path, file_name))
    plt.close()


def train_or_load_mipro_model(data, model, model_name):
    if os.path.isfile(os.path.join('tpsa_saved_data', f'{model_name[0:10]}.json')):
        load_path = os.path.join('tpsa_saved_data', f'{model_name[0:10]}.json')
        mipro_model = model
        mipro_model.load(path=load_path)
    else:
        examples = create_examples(data)
        mipro_model = train_mipro_model(model, examples, model_name)
        save_path = os.path.join('tpsa_saved_data', f'{model_name}.json')
        mipro_model.save(save_path)
    return mipro_model


def load_or_split_data(small_sample, large_sample):
    try:
        train_df = pd.read_pickle(os.path.join('tpsa_saved_data', 'tpsa_train_df'))
        test_df = pd.read_pickle(os.path.join('tpsa_saved_data', 'tpsa_test_df'))

    except:
        file_path = os.path.join('tpsa_saved_data', 'balanced_tpsa_data.csv')
        data = load_pubchem_data(file_path)
        data.drop(data.loc[data['Charge'] != 0].index, inplace=True)
        data.drop(data.loc[data['HBondAcceptorCount'] > 10].index, inplace=True)#Lipinsky rule
        data.drop(data.loc[data['HBondDonorCount'] > 5].index, inplace=True)#Lipinsky rule
        data['ExactMass'] = pd.to_numeric(data['ExactMass'], errors='coerce')
        data.drop(data.loc[data['ExactMass'] > 500].index, inplace=True)#Lipinsky rule

        # Convert the data into format with SMILES in the ids field
        X = np.zeros(len(data))
        y = data['TPSA'].values
        smiles_list = data['CanonicalSMILES'].values

        # Create a DeepChem DiskDataset
        dataset = dc.data.DiskDataset.from_numpy(X, y, ids=smiles_list)

        # Use ScaffoldSplitter to split the dataset into train/test sets
        splitter = dc.splits.ScaffoldSplitter()
        train_dataset, test_dataset = splitter.train_test_split(dataset)

        # Extract train and test indices
        train_smiles = train_dataset.ids
        test_smiles = test_dataset.ids

        # Create train and test dataframes
        train_df = data[data['CanonicalSMILES'].isin(train_smiles)].reset_index(drop=True)
        test_df = data[data['CanonicalSMILES'].isin(test_smiles)].reset_index(drop=True)

        train_df = train_df.sample(large_sample)
        test_df = test_df.sample(small_sample)

        train_df.to_pickle(os.path.join('tpsa_saved_data', 'tpsa_train_df'))
        test_df.to_pickle(os.path.join('tpsa_saved_data', 'tpsa_test_df'))

    return train_df, test_df


def train_mipro_model(model, examples, model_name):

    teleprompter = MIPROv2(prompt_model=turbo,
                           task_model=turbo,
                           max_errors=10,
                           num_candidates=10,
                           num_threads=6,
                           verbose=True,
                           metric=tpsa_match_metric,
                           log_dir=os.path.join('tpsa_saved_data'),
                           init_temperature=1.2,
                           track_stats=True,
                           )
    mipro_tpsa_program = teleprompter.compile(model,
                                              trainset=examples[0:29],
                                              valset=examples[30:],
                                              num_trials=25,
                                              minibatch=True,
                                              minibatch_size=5,
                                              program_aware_proposer=False,
                                              max_bootstrapped_demos=8,
                                              requires_permission_to_run=False,
                                              max_labeled_demos=8,
                                              minibatch_full_eval_steps=5,
                                              )

    save_path = os.path.join('tpsa_saved_data', f'{model_name}.json')
    mipro_tpsa_program.save(save_path)

    return mipro_tpsa_program


def create_sample_data(model_name, model, training_data):
    save_path = 'tpsa_saved_data'
    predictions = []

    for smiles in training_data['CanonicalSMILES']:
        calculated_value = training_data[training_data['CanonicalSMILES'] == smiles]['TPSA'].values[0]

        # Depending on the model type, use the correct prediction function
        if model_name == "direct_model":
            prediction = predict_tpsa_straight(smiles)
            predictions.append({
                'smiles': smiles,
                'calculated': float(calculated_value),
                f'{model_name}_pred': prediction,
            })
        else:
            prediction, group_descriptions, num_n, num_o, data_table = model.forward(question=smiles, answer='guess')
            try:
                prediction = float(prediction.answer.tpsa_numbers_list)
            except:
                prediction = float(prediction[1].tpsa_numbers_list)  # Assuming the output format
            predictions.append({
                'smiles': smiles,
                'calculated': float(calculated_value),
                f'{model_name}_pred': prediction,
                'group_descriptions': group_descriptions,
                'num_n': num_n,
                'num_o': num_o,
            })

    predictions_df = pd.DataFrame(predictions)
    file_name = os.path.join(save_path, f'{model_name}_predictions.csv')
    predictions_df.to_csv(file_name, index=False)
    print(f'Saved predictions for {model_name} model to {file_name}')


def run_model(model, model_name):
    first_split, second_split = load_or_split_data(60, 140)
    if model_name != 'direct_model':
        train_or_load_mipro_model(second_split, model, model_name)
    # create_sample_data(model_name, model, first_split) # un-comment to create datasets

    file_path = os.path.join('tpsa_saved_data')
    file_name = os.path.join(file_path, f'{model_name}_predictions.csv')
    predictions_df = pd.read_csv(file_name)
    create_plot('calculated', f'{model_name}_pred', 'o', f'{model_name}.png', predictions_df,
                'results', model_name)


def main():
    dspy.settings.configure(lm=turbo, rm=turbo)
    file_path = os.path.join('tpsa_saved_data', 'tpsa_smarts.csv')
    smarts_input = pd.read_csv(file_path, index_col=0, header=0,
                               names=["Index", "Original_SMARTS", "Functional_Group_Name", "New_SMARTS_1",
                                      "New_SMARTS_2", "New_SMARTS_3"])
    global functional_groups
    global descriptive_tpsa_data
    functional_groups = load_smarts_patterns(smarts_input)
    file_path = os.path.join('tpsa_saved_data', 'tpsa_values.csv')
    descriptive_tpsa_data = pd.read_csv(file_path)

    MIPRO_model = MiproTPSA()
    models = ['tpsa_model_acf', 'direct_model', 'tpsa_model', 'tpsa_model_abcdf', 'tpsa_model_acdef',
              'tpsa_model_no_demos', 'tpsa_model_no_sig']
    for model in models:
        run_model(MIPRO_model, model)


if __name__ == "__main__":
    main()
