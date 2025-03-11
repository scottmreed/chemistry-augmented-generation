import os
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem import Descriptors
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind


def find_longest_conjugated_sequence(mol):
    if mol is None:
        return 0

    longest_sequence = 0
    current_sequence = 0
    conjugated_bonds = []

    for bond in mol.GetBonds():
        if bond.GetIsConjugated():
            conjugated_bonds.append(bond)

    if not conjugated_bonds:
        return 0

    # Iterate through all bonds
    for i, bond in enumerate(conjugated_bonds):
        if bond.GetIsConjugated():
            current_sequence += 1
            if i < len(conjugated_bonds) - 1:
                # Check if the next bond is connected to the current bond
                next_bond = conjugated_bonds[i + 1]
                if next_bond.GetBeginAtom().GetIdx() not in [bond.GetBeginAtom().GetIdx(),
                                                             bond.GetEndAtom().GetIdx()] and next_bond.GetEndAtom().GetIdx() not in [
                    bond.GetBeginAtom().GetIdx(), bond.GetEndAtom().GetIdx()]:
                    if current_sequence > longest_sequence:
                        longest_sequence = current_sequence
                    current_sequence = 0
            else:
                if current_sequence > longest_sequence:
                    longest_sequence = current_sequence
        else:
            if current_sequence > longest_sequence:
                longest_sequence = current_sequence
            current_sequence = 0

    return longest_sequence


def analyze_tpsa_differences(file_path, model, width):
    # Read the CSV data
    descriptive_tpsa_data = pd.read_csv(file_path)

    # Add a new column for delta TPSA
    descriptive_tpsa_data['delta_tpsa'] = abs(
        descriptive_tpsa_data['calculated'] - descriptive_tpsa_data[f'{model}_pred'])

    # Separate data into two groups based on delta TPSA threshold of > 1
    group_greater_1 = descriptive_tpsa_data[descriptive_tpsa_data['delta_tpsa'] > 1]
    group_less_equal_1 = descriptive_tpsa_data[descriptive_tpsa_data['delta_tpsa'] <= 1]

    print(f'model is {model}')

    if model != 'direct_model':
        print(f'groups present are: ', descriptive_tpsa_data['group_descriptions'])
        print("Statistics for molecules with delta TPSA <= 1:")

        # Generate RDKit molecules and corresponding delta TPSA values for visualization
        mols = []
        legends = []

        for index, row in group_greater_1.iterrows():
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

    return descriptive_tpsa_data, group_greater_1, group_less_equal_1

def calculate_rotatable_bonds(mol):
    if mol is None:
        return 0
    return Descriptors.NumRotatableBonds(mol)

models = {'tpsa_model_abcdef': 10, 'tpsa_model_abcdf': 10, 'tpsa_model_acdef': 10, 'tpsa_model_acf': 10,
          'tpsa_model_abcdef_no_demos': 10, 'tpsa_model_abcdef_no_sig': 10, 'direct_model': 0}
all_models_data = {}
outlier_data = {}
good_data = {}
for llm_model, width in models.items():
    file_name = f'{llm_model}_predictions.csv'
    csv_path = os.path.join('tpsa_saved_data', file_name)
    all_models_data[llm_model], outlier_data[llm_model], good_data[llm_model] = analyze_tpsa_differences(csv_path,
                                                                                                         llm_model,
                                                                                                         width)

# Data collection for bar graphs
model_names = []
mean_conjugation_lengths = []
std_conjugation_lengths = []
mean_rotatable_bonds = []
std_rotatable_bonds = []
mean_num_n_atoms = []
std_num_n_atoms = []
mean_num_o_atoms = []
std_num_o_atoms = []

models['tpsa_model_abcdef (All)'] = 0
ordered_models = [m for m in models if m != 'direct_model']

for llm_model in ordered_models:
    width = models[llm_model]
    longest_conjugated_sequences = []
    rotatable_bonds_counts = []

    if llm_model == 'tpsa_model_abcdef (All)':
        model_names.append(llm_model)
        df = all_models_data['tpsa_model_abcdef']
        file_name = f'tpsa_model_abcdef_predictions.csv'
        csv_path = os.path.join('tpsa_saved_data', file_name)
        descriptive_tpsa_data = pd.read_csv(csv_path)
        longest_conjugated_sequences = []
        rotatable_bonds_counts = []

        for smiles in df['smiles']:
            mol = Chem.MolFromSmiles(smiles)
            longest_sequence = find_longest_conjugated_sequence(mol)
            longest_conjugated_sequences.append(longest_sequence)

            rotatable_bonds = calculate_rotatable_bonds(mol)
            rotatable_bonds_counts.append(rotatable_bonds)

        print(f"\nConjugated Bond Analysis for full {llm_model}:")
        print(f"  Full Data: Mean longest contiguous conjugated bonds: {np.mean(longest_conjugated_sequences):.2f}")
        print(f"  Full Data: STD longest contiguous conjugated bonds: {np.std(longest_conjugated_sequences):.2f}")

        print(f"\nRotatable Bond Analysis for full {llm_model}:")
        print(f"  Full Data: Mean number of rotatable bonds: {np.mean(rotatable_bonds_counts):.2f}")
        print(f"  Full Data: STD number of rotatable bonds: {np.std(rotatable_bonds_counts):.2f}")

        mean_conjugation_lengths.append(np.mean(longest_conjugated_sequences))
        std_conjugation_lengths.append(np.std(longest_conjugated_sequences))
        mean_rotatable_bonds.append(np.mean(rotatable_bonds_counts))
        std_rotatable_bonds.append(np.std(rotatable_bonds_counts))
        mean_num_n_atoms.append(all_models_data['tpsa_model_abcdef']['num_n'].mean())
        std_num_n_atoms.append(all_models_data['tpsa_model_abcdef']['num_n'].std())
        mean_num_o_atoms.append(all_models_data['tpsa_model_abcdef']['num_o'].mean())
        std_num_o_atoms.append(all_models_data['tpsa_model_abcdef']['num_o'].std())

    else:
        df = outlier_data[llm_model]

        for smiles in df['smiles']:
            mol = Chem.MolFromSmiles(smiles)
            longest_sequence = find_longest_conjugated_sequence(mol)
            longest_conjugated_sequences.append(longest_sequence)

            rotatable_bonds = calculate_rotatable_bonds(mol)
            rotatable_bonds_counts.append(rotatable_bonds)

        print(f"\nConjugated Bond Analysis for {llm_model}:")
        print(f"  Mean longest contiguous conjugated bonds: {np.mean(longest_conjugated_sequences):.2f}")
        print(f"  STD longest contiguous conjugated bonds: {np.std(longest_conjugated_sequences):.2f}")

        print(f"\nRotatable Bond Analysis for {llm_model}:")
        print(f"  Mean number of rotatable bonds: {np.mean(rotatable_bonds_counts):.2f}")
        print(f"  STD number of rotatable bonds: {np.std(rotatable_bonds_counts):.2f}")

        model_names.append(llm_model + r" ($\Delta$TPSA > 1)")

        mean_conjugation_lengths.append(np.mean(longest_conjugated_sequences))
        std_conjugation_lengths.append(np.std(longest_conjugated_sequences))
        mean_rotatable_bonds.append(np.mean(rotatable_bonds_counts))
        std_rotatable_bonds.append(np.std(rotatable_bonds_counts))
        # n and o count for outliers
        outlier_n_and_o = outlier_data[llm_model]
        mean_num_n_atoms.append(outlier_n_and_o['num_n'].mean())
        std_num_n_atoms.append(outlier_n_and_o['num_n'].std())
        mean_num_o_atoms.append(outlier_n_and_o['num_o'].mean())
        std_num_o_atoms.append(outlier_n_and_o['num_o'].std())

# Compute counts for each model from good_data and outlier_data
model_labels = []
good_counts = []
outlier_counts = []
total_counts = []

for llm_model in ordered_models:
    # Get counts from the dictionaries
    if llm_model == 'tpsa_model_abcdef (All)':
        continue
    good_count = len(good_data[llm_model])
    outlier_count = len(outlier_data[llm_model])
    total = good_count + outlier_count

    # Append labels and counts
    model_labels.append(llm_model)
    good_counts.append(good_count)
    outlier_counts.append(outlier_count)
    total_counts.append(total)


# --- Compute means and t-test p-values for each property ---
# These lists will store the mean values and p-values for each model.
good_conjugation = []
outlier_conjugation = []
p_conjugation = []

good_rotatable = []
outlier_rotatable = []
p_rotatable = []

good_n_atoms = []
outlier_n_atoms = []
p_n_atoms = []

good_o_atoms = []
outlier_o_atoms = []
p_o_atoms = []

model_labels = []
ordered_models = [m for m in models if m != 'direct_model' and m != 'tpsa_model_abcdef (All)']

for model in ordered_models:
    label = model.split(" ($")[0] if "($" in model else model
    model_labels.append(label)

    key = model  # key used in good_data and outlier_data
    df_good = good_data[key]
    df_outlier = outlier_data[key]

    # --- Conjugation ---
    conj_good = []
    for smiles in df_good['smiles']:
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            conj_good.append(find_longest_conjugated_sequence(mol))
    conj_out = []
    for smiles in df_outlier['smiles']:
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            conj_out.append(find_longest_conjugated_sequence(mol))
    mean_conj_good = np.mean(conj_good) if conj_good else 0
    mean_conj_out = np.mean(conj_out) if conj_out else 0
    good_conjugation.append(mean_conj_good)
    outlier_conjugation.append(mean_conj_out)
    if conj_good and conj_out:
        _, p_val = ttest_ind(conj_good, conj_out, equal_var=False)
    p_conjugation.append(p_val)

    # --- Rotatable Bonds ---
    rot_good = []
    for smiles in df_good['smiles']:
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            rot_good.append(calculate_rotatable_bonds(mol))
    rot_out = []
    for smiles in df_outlier['smiles']:
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            rot_out.append(calculate_rotatable_bonds(mol))
    mean_rot_good = np.mean(rot_good) if rot_good else 0
    mean_rot_out = np.mean(rot_out) if rot_out else 0
    good_rotatable.append(mean_rot_good)
    outlier_rotatable.append(mean_rot_out)
    if rot_good and rot_out:
        _, p_val = ttest_ind(rot_good, rot_out, equal_var=False)
    p_rotatable.append(p_val)

    # --- N Atoms ---
    n_good = df_good['num_n'].dropna().values
    n_out = df_outlier['num_n'].dropna().values
    mean_n_good = np.mean(n_good) if len(n_good) > 0 else 0
    mean_n_out = np.mean(n_out) if len(n_out) > 0 else 0
    good_n_atoms.append(mean_n_good)
    outlier_n_atoms.append(mean_n_out)
    if len(n_good) > 0 and len(n_out) > 0:
        _, p_val = ttest_ind(n_good, n_out, equal_var=False)
    p_n_atoms.append(p_val)

    # --- O Atoms ---
    o_good = df_good['num_o'].dropna().values
    o_out = df_outlier['num_o'].dropna().values
    mean_o_good = np.mean(o_good) if len(o_good) > 0 else 0
    mean_o_out = np.mean(o_out) if len(o_out) > 0 else 0
    good_o_atoms.append(mean_o_good)
    outlier_o_atoms.append(mean_o_out)
    if len(o_good) > 0 and len(o_out) > 0:
        _, p_val = ttest_ind(o_good, o_out, equal_var=False)
    p_o_atoms.append(p_val)

# --- Plotting parameters for grouped bar charts ---
n_models = len(model_labels)
x = np.arange(n_models)
bar_width = 0.35
text_fontsize = 14
fig, axs = plt.subplots(2, 2, figsize=(12, 12), dpi=300)

# --- Conjugation Length (Top Left) ---
ax = axs[0, 0]
ax.bar(x - bar_width/2, good_conjugation, width=bar_width,
       color='white', edgecolor='black', label='TPSA ≤ 1')
ax.bar(x + bar_width/2, outlier_conjugation, width=bar_width,
       color='black', edgecolor='black', label='TPSA > 1')
for i in range(n_models):
    if p_conjugation[i] < 0.05:
         max_val = max(good_conjugation[i], outlier_conjugation[i])
         offset = 0.05 * max_val  # 5% above the taller bar
         ax.text(x[i], max_val + offset, '*', ha='center', va='bottom',
                 fontsize=text_fontsize, color='black')
ax.set_xticks(x)
ax.set_xticklabels(model_labels, rotation=45, ha='right', fontsize=text_fontsize)
ax.set_xlabel("Model", fontsize=text_fontsize)
ax.set_ylabel("Mean Conjugation Length", fontsize=text_fontsize)
ax.set_title("Conjugation Length by TPSA Category", fontsize=text_fontsize)
ax.text(-0.19, 1.05, "A", transform=ax.transAxes, fontsize=text_fontsize)
max_val_all = max(good_conjugation + outlier_conjugation)
ax.set_ylim(0, max_val_all + 1)
ax.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=text_fontsize)

# --- Rotatable Bonds (Top Right) ---
ax = axs[0, 1]
ax.bar(x - bar_width/2, good_rotatable, width=bar_width,
       color='white', edgecolor='black', label='TPSA ≤ 1')
ax.bar(x + bar_width/2, outlier_rotatable, width=bar_width,
       color='black', edgecolor='black', label='TPSA > 1')
for i in range(n_models):
    if p_rotatable[i] < 0.05:
         max_val = max(good_rotatable[i], outlier_rotatable[i])
         offset = 0.05 * max_val
         ax.text(x[i], max_val + offset, '*', ha='center', va='bottom',
                 fontsize=text_fontsize, color='black')
ax.set_xticks(x)
ax.set_xticklabels(model_labels, rotation=45, ha='right', fontsize=text_fontsize)
ax.set_xlabel("Model", fontsize=text_fontsize)
ax.set_ylabel("Mean Rotatable Bonds", fontsize=text_fontsize)
ax.set_title("Rotatable Bonds by TPSA Category", fontsize=text_fontsize)
ax.text(-0.19, 1.05, "B", transform=ax.transAxes, fontsize=text_fontsize)
ax.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=text_fontsize)

# --- N Atoms (Bottom Left) ---
ax = axs[1, 0]
ax.bar(x - bar_width/2, good_n_atoms, width=bar_width,
       color='white', edgecolor='black', label='TPSA ≤ 1')
ax.bar(x + bar_width/2, outlier_n_atoms, width=bar_width,
       color='black', edgecolor='black', label='TPSA > 1')
for i in range(n_models):
    if p_n_atoms[i] < 0.05:
         max_val = max(good_n_atoms[i], outlier_n_atoms[i])
         offset = 0.05 * max_val
         ax.text(x[i], max_val + offset, '*', ha='center', va='bottom',
                 fontsize=text_fontsize, color='black')
ax.set_xticks(x)
ax.set_xticklabels(model_labels, rotation=45, ha='right', fontsize=text_fontsize)
ax.set_xlabel("Model", fontsize=text_fontsize)
ax.set_ylabel("Mean N Atoms", fontsize=text_fontsize)
ax.set_title("N Atoms by TPSA Category", fontsize=text_fontsize)
ax.text(-0.19, 1.05, "C", transform=ax.transAxes, fontsize=text_fontsize)
ax.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=text_fontsize)

# --- O Atoms (Bottom Right) ---
ax = axs[1, 1]
ax.bar(x - bar_width/2, good_o_atoms, width=bar_width,
       color='white', edgecolor='black', label='TPSA ≤ 1')
ax.bar(x + bar_width/2, outlier_o_atoms, width=bar_width,
       color='black', edgecolor='black', label='TPSA > 1')
for i in range(n_models):
    if p_o_atoms[i] < 0.05:
         max_val = max(good_o_atoms[i], outlier_o_atoms[i])
         offset = 0.05 * max_val
         ax.text(x[i], max_val + offset, '*', ha='center', va='bottom',
                 fontsize=text_fontsize, color='black')
ax.set_xticks(x)
ax.set_xticklabels(model_labels, rotation=45, ha='right', fontsize=text_fontsize)
ax.set_xlabel("Model", fontsize=text_fontsize)
ax.set_ylabel("Mean O Atoms", fontsize=text_fontsize)
ax.set_title("O Atoms by TPSA Category", fontsize=text_fontsize)
ax.text(-0.19, 1.05, "D", transform=ax.transAxes, fontsize=text_fontsize)

max_val_all = max(good_o_atoms + outlier_o_atoms)
ax.set_ylim(0, max_val_all + 1)
ax.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=text_fontsize)

plt.tight_layout()
plt.savefig("results/grouped_properties.png", dpi=300)
plt.show()
