import pandas as pd
import requests
import time
import random
import os
from rdkit import Chem


file_path = os.path.join('tpsa_saved_data', 'tpsa_smarts.csv')
smarts_input = pd.read_csv(file_path, index_col=0, header=0, names=["Index", "Initial_SMARTS", "Functional_Group_Name",
                                                                    "New_SMARTS_1", "New_SMARTS_2", "New_SMARTS_3"])


def load_smarts_patterns(df):
    """
    Create a dictionary mapping functional group names to lists of RDKit Mol objects
    from the SMARTS patterns, including original and additional patterns.

    Parameters:
    - df (pandas.DataFrame): DataFrame containing SMARTS patterns with columns:
        'Original_SMARTS', 'Functional_Group_Name', 'New_SMARTS_1', 'New_SMARTS_2', 'New_SMARTS_3'

    Returns:
    - functional_groups (dict): Dictionary where keys are functional group names and
                                values are lists of RDKit Mol objects representing SMARTS patterns.
    """
    functional_groups = {}

    # Iterate over each row in the DataFrame
    for _, row in df.iterrows():
        # Extract the functional group name
        group_name = row["Functional_Group_Name"]

        # Initialize a list to hold all SMARTS patterns for this functional group
        smarts_patterns = []

        # Add the Original_SMARTS pattern
        original_smarts = row.get("Original_SMARTS")
        if pd.notnull(original_smarts) and original_smarts.strip():
            smarts_patterns.append(original_smarts.strip())

        # Add all New_SMARTS_* patterns
        for i in range(1, 4):
            new_smarts = row.get(f"New_SMARTS_{i}")
            if pd.notnull(new_smarts) and new_smarts.strip():
                smarts_patterns.append(new_smarts.strip())

        # Initialize a list to hold valid RDKit Mol objects for the current group
        mol_patterns = []

        # Validate and convert each SMARTS pattern to an RDKit Mol object
        for smarts in smarts_patterns:
            mol = Chem.MolFromSmarts(smarts)
            if mol is None:
                print(f"Invalid SMARTS pattern: '{smarts}' for group: '{group_name}'. Skipping this pattern.")
            else:
                mol_patterns.append(mol)

        # If there are valid Mol objects, add them to the dictionary
        if mol_patterns:
            if group_name not in functional_groups:
                functional_groups[group_name] = []
            functional_groups[group_name].extend(mol_patterns)
        else:
            print(f"No valid SMARTS patterns found for functional group: '{group_name}'.")

    return functional_groups


functional_group_list = load_smarts_patterns(smarts_input)
file_path = os.path.join('tpsa_saved_data', 'tpsa_values.csv')
descriptive_tpsa_data = pd.read_csv(file_path)

def describe_molecule(smiles, functional_groups):
    """
    Takes a SMILES code and returns a structured description of the molecule by assigning
    each Nitrogen (N) and Oxygen (O) atom to one and only one functional group based on loaded SMARTS patterns.

    The function performs the following steps:
    1. Scans all original SMARTS patterns for matches.
    2. If no original SMARTS matches a group, scans additional SMARTS patterns.
    3. Assigns each group to one set of atoms, ensuring no atom is part of multiple groups.
    4. Reports errors and unassigned atoms but only returns the "Assigned Functional Groups".

    Parameters:
    - smiles (str): The SMILES string of the molecule.
    - functional_groups (dict): Dictionary where keys are functional group names and
                                values are lists of RDKit Mol objects representing SMARTS patterns.
                                The first pattern in each list is considered the original SMARTS.

    Returns:
    - dict: A dictionary containing only "Assigned Functional Groups".
    """
    # Convert SMILES to RDKit molecule
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        print("Error: Invalid SMILES code.")
        return {}

    # Initialize tracking structures
    assigned_atoms = set()  # Atoms that have been assigned to a functional group
    assigned_groups = []  # List of assigned functional groups with atom indices
    errors = []  # List of errors (multiple assignments)
    unassigned_atoms = []  # List of unassigned target atoms

    # Define target atom symbols
    target_symbols = {'N', 'O'}

    # Iterate through functional groups in the order they appear
    for group_name, patterns in functional_groups.items():
        if not patterns:
            continue  # Skip if no patterns are available

        # Assume the first pattern is the original SMARTS
        original_pattern = patterns[0]
        additional_patterns = patterns[1:]

        # Step 1: Scan Original SMARTS Patterns
        matches = mol.GetSubstructMatches(original_pattern, uniquify=True)
        for match in matches:
            # Check if any atom in the match is already assigned
            overlap = assigned_atoms.intersection(match)
            if overlap:
                overlapping_groups = [grp['Functional_Group'] for grp in assigned_groups
                                      if any(atom in overlap for atom in grp['Atom_Indices'])]
                if group_name not in overlapping_groups:
                    return None

                continue  # Skip assigning this match to avoid double-counting

            # Assign the group to all atoms in the match
            assigned_groups.append({
                "Functional_Group": group_name,
                "Atom_Indices": list(match)
            })
            assigned_atoms.update(match)

        # Step 2: Scan Additional SMARTS Patterns for Unassigned Atoms
        for add_pattern in additional_patterns:
            if add_pattern is None:
                continue  # Skip if the pattern is None
            matches = mol.GetSubstructMatches(add_pattern, uniquify=True)
            for match in matches:
                # Check if any atom in the match is already assigned
                overlap = assigned_atoms.intersection(match)
                if overlap:
                    overlapping_groups = [grp['Functional_Group'] for grp in assigned_groups
                                          if any(atom in overlap for atom in grp['Atom_Indices'])]
                    if group_name not in overlapping_groups:
                        # return None
                        # Assign the group to all atoms in the match
                        assigned_groups.append({
                            "Functional_Group": group_name,
                            "Atom_Indices": list(match)
                        })
                        assigned_atoms.update(match)
                    continue  # Skip assigning this match to avoid double-counting
                else:
                    assigned_groups.append({
                        "Functional_Group": group_name,
                        "Atom_Indices": list(match)
                    })
                    assigned_atoms.update(match)

    # Step 3: Identify Unassigned Target Atoms
    for atom in mol.GetAtoms():
        if atom.GetSymbol() in target_symbols:
            atom_idx = atom.GetIdx()
            if atom_idx not in assigned_atoms:
                print('N or O atom unassigned')

    # Step 4: Report Errors and Unassigned Atoms
    if errors:
        print("--- Errors ---")
        for error in errors:
            grp = error["Group"]
            overlapping = ", ".join(error["Overlapping_Functional_Groups"])
            atoms = ", ".join(map(str, error["Atom_Indices"]))
            print(f"Functional Group '{grp}' overlaps with existing groups '{overlapping}' on atoms {atoms}.")

    if unassigned_atoms:
        print("--- Unassigned Atoms ---")
        for atom in unassigned_atoms:
            idx = atom["Atom_Index"]
            symbol = atom["Atom_Symbol"]
            issue = atom["Issue"]
            print(f"Atom {idx} ({symbol}) could not be assigned to any functional group. Issue: {issue}")


    # Step 5: Prepare and Return Assigned Functional Groups
    description = {}
    if assigned_groups:
        print(f"Functional Group {assigned_groups}")
        description["Assigned Functional Groups"] = [
            f"{grp['Functional_Group']}"
            for grp in assigned_groups
        ]
    else:
        description["Assigned Functional Groups"] = ["No functional groups assigned."]

    return description


def is_composed_of_CNO(smiles: str) -> bool:
    """
    Determines whether a molecule, represented by a SMILES string, is composed exclusively
    of Carbon (C), Nitrogen (N), and Oxygen (O) atoms.

    Parameters:
    - smiles (str): The SMILES string of the molecule to be evaluated.

    Returns:
    - bool:
        - True if all heavy atoms in the molecule are C, N, or O.
        - False otherwise or if the SMILES string is invalid.
    """
    try:
        # Convert SMILES string to RDKit molecule object
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            # Invalid SMILES string
            return False

        # Iterate over all heavy atoms in the molecule
        for atom in mol.GetAtoms():
            # Get the atomic symbol (e.g., 'C', 'N', 'O', etc.)
            symbol = atom.GetSymbol()
            if symbol not in {'C', 'N', 'O'}:
                # Found an atom that is not C, N, or O
                return False

        # All heavy atoms are C, N, or O
        return True

    except Exception as e:
        return False


def get_highest_cid(starting_cid=138962044, chunks=100):
    max_cid = starting_cid
    total_checked = 0

    for chunk in range(chunks):
        # Calculate the starting point for the current chunk
        start = starting_cid + (chunk * 100)
        url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{start}/JSON"

        response = requests.get(url)

        if response.status_code == 200:
            data = response.json()
            if 'PC_Compounds' in data:
                for compound in data['PC_Compounds']:
                    cid = compound['id']['id']['cid']
                    if cid > max_cid:
                        max_cid = cid
                total_checked += len(data['PC_Compounds'])
            else:
                print(f"No more compounds found starting from CID {start}.")
                break
        else:
            print(f"Error fetching data from PubChem for starting CID {start}.")
            break

    print(f"The highest CID number found is: {max_cid}")
    print(f"Total compounds checked: {total_checked}")
    return max_cid


# Function to fetch properties for a list of CIDs using smaller chunks
def fetch_pubchem_properties(cids, chunk_size=10):
    """Fetch properties for a list of CIDs from PubChem using GET requests."""
    molecule_data = []

    for i in range(0, len(cids), chunk_size):
        cid_chunk = cids[i:i + chunk_size]
        cid_list = ",".join(map(str, cid_chunk))  # Create comma-separated list of CIDs
        property_url_chunk = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid_list}/property/XLogP,ExactMass,Charge,HBondDonorCount,HBondAcceptorCount,TPSA,CanonicalSMILES/JSON"

        response = requests.get(property_url_chunk)
        if response.status_code == 200:
            # Extract properties and add to molecule_data
            properties = response.json().get('PropertyTable', {}).get('Properties', [])
            molecule_data.extend(properties)
        else:
            print(f"Error fetching data for CIDs: {cid_list}. HTTP Status: {response.status_code}")

        time.sleep(1)

    return molecule_data


def main():
    # Define the highest CID value for reference
    cid_max = get_highest_cid(starting_cid=138960000, chunks=100)

    bins = list(range(15, 75, 5))
    bin_count_target = 20

    # Dictionary to track molecules in each bin
    bin_counts = {bin_value: 0 for bin_value in bins}

    molecule_data = []

    while not all([count >= bin_count_target for count in bin_counts.values()]):
        # Generate a random set of 100 CIDs
        random_cids = [random.randint(1, cid_max) for _ in range(100)]

        # Fetch properties using smaller GET chunks
        properties = fetch_pubchem_properties(random_cids, chunk_size=10)

        # Filter out entries without a TPSA value
        valid_properties = [prop for prop in properties if 'TPSA' in prop]

        # Assign molecules to bins based on their TPSA values
        for prop in valid_properties:
            tpsa_value = prop['TPSA']
            # Determine the bin for this TPSA value
            for bin_value in bins:
                if bin_value <= tpsa_value < bin_value + 5:
                    # Check if this bin is still below the target count
                    if bin_counts[bin_value] < bin_count_target:
                        if is_composed_of_CNO(prop['CanonicalSMILES']):
                            group_check = describe_molecule(prop['CanonicalSMILES'], functional_group_list)
                            if group_check:
                                molecule_data.append(prop)  # Add molecule to the collection
                                bin_counts[bin_value] += 1  # Increment bin count
                                # Print update if bin is completed
                                if bin_counts[bin_value] == bin_count_target:
                                    print(
                                        f"Bin {bin_value} to {bin_value + 5} completed with {bin_counts[bin_value]} molecules.")
                    break  # Move to next molecule once assigned to a bin

    df_balanced = pd.DataFrame(molecule_data)
    output_dir = 'tpsa_saved_data'

    output_path = os.path.join(output_dir, 'balanced_tpsa_data.csv')
    df_balanced.to_csv(output_path, index=False)

    print(f"Data saved to {output_path}")

if __name__ == "__main__":
    main()
