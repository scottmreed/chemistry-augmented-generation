import os
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Draw


def analyze_tpsa_differences(file_path, model, width):
    # Read the CSV data
    descriptive_tpsa_data = pd.read_csv(file_path)

    # Add a new column for delta TPSA
    descriptive_tpsa_data['delta_tpsa'] = abs(
        descriptive_tpsa_data['calculated'] - descriptive_tpsa_data[f'{model}_pred'])

    # Separate data into two groups based on delta TPSA threshold of > 1
    group_greater_5 = descriptive_tpsa_data[descriptive_tpsa_data['delta_tpsa'] > 5]
    group_less_equal_5 = descriptive_tpsa_data[descriptive_tpsa_data['delta_tpsa'] <= 5]

    print(f'model is {model}')
    print("Statistics for molecules with delta TPSA <= 1:")

    if model != 'direct_model':
        # Calculate statistics for the sum of num_n and num_o
        group_greater_5_sum = group_greater_5['num_n'] + group_greater_5['num_o']
        group_less_equal_5_sum = group_less_equal_5['num_n'] + group_less_equal_5['num_o']
        print(f"  Average sum of num_n and num_o for molecules with delta TPSA <= 5: {group_less_equal_5_sum.mean()}")
        print(f"  Average sum of num_n and num_o for molecules with delta TPSA > 5: {group_greater_5_sum.mean()}")

        # Calculate the length of group descriptions
        group_greater_5_len_desc = group_greater_5['group_descriptions'].apply(
            lambda x: len(x.split()) if isinstance(x, str) else 0)
        group_less_equal_5_len_desc = group_less_equal_5['group_descriptions'].apply(
            lambda x: len(x.split()) if isinstance(x, str) else 0)

        print(f"  Average length of group descriptions for molecules with delta TPSA <= 5: {group_less_equal_5_len_desc.mean()}\n")
        print(f"  Total number of outliers: {len(group_greater_5_len_desc)}\n")
        print(f"  Average length of group descriptions for molecules with delta TPSA > 5: {group_greater_5_len_desc.mean()}\n")

        # Generate RDKit molecules and corresponding delta TPSA values for visualization
        mols = []
        legends = []

        for index, row in group_greater_5.iterrows():
            smiles = row['smiles']  # Assuming there is a 'smiles' column with the molecule SMILES string
            calculated = row['calculated']
            predicted = row[f'{model}_pred']
            delta_tpsa = abs(round(calculated - predicted, 2))

            mol = Chem.MolFromSmiles(smiles)
            if mol:
                mols.append(mol)
                legends.append(f"TPSA difference: {delta_tpsa}")

        # Create a grid image of the molecules
        if mols:
            img = Draw.MolsToGridImage(mols, legends=legends, molsPerRow=width, subImgSize=(500, 500))
            img.save(f'results/{model}_outliers.png')
        else:
            print("No molecules found with a TPSA difference greater than 5.")


models = {'tpsa_model_acf': 10, 'tpsa_model_abcdef': 10, 'tpsa_model_abcdf': 10, 'tpsa_model_acdef': 10,
          'tpsa_model_abcdef_no_demos': 10, 'tpsa_model_abcdef_no_sig': 10, 'direct_model': 0}

for llm_model, width in models.items():
    file_name = f'{llm_model}_predictions.csv'
    csv_path = os.path.join('tpsa_saved_data', file_name)
    analyze_tpsa_differences(csv_path, llm_model, width)
