from rdkit import Chem
import pandas as pd
import os


def test_smarts_patterns(descriptions):
    """Function to validate if the SMARTS patterns in the descriptions dictionary are valid."""
    invalid_smarts = []  # List to hold invalid SMARTS patterns

    # Iterate through the dictionary and check each SMARTS pattern
    for smarts, description in descriptions.items():
        mol = Chem.MolFromSmarts(smarts)  # Attempt to create an RDKit molecule from SMARTS
        if mol is None:
            print(f"Invalid SMARTS: {smarts} -> {description}")
            invalid_smarts.append((smarts, description))
        else:
            print(f"Valid SMARTS: {smarts} -> {description}")

    # Report the number of invalid SMARTS patterns found
    if invalid_smarts:
        print(f"\n{len(invalid_smarts)} invalid SMARTS patterns found:")
        for smarts, description in invalid_smarts:
            print(f"  {smarts} -> {description}")
    else:
        print("\nAll SMARTS patterns are valid!")


group_descriptions = {
    "[N]([R1])([R2])[R3]": "Tertiary amine",
    "[nH](:[!H]):[!H]": "Aromatic nitrogen with hydrogen",
    "[N]([!H])=[!H]": "Imine",
    "[n+](:[!H])(:[!H]):[!H]": "Aromatic Charged nitrogen",
    "[N]#[!H]": "Nitrile group",
    "[n+]([!H]):[!H]:[!H]": "Aromatic Charged nitrogen",
    "[N](=O)(=O)": "Nitro group",
    "[nH+](:[!H]):[!H]": "Aromatic nitrogen with hydrogen and positive charge",
    "[N]#N": "Azide group (middle nitrogen)",
    "[O]([!H])[!H]": "Ether",
    "[N]1([!H])[!H][!H][!H]1": "Three-membered ring nitrogen",
    "[O]1[!H][!H][!H]1": "Three-membered ring oxygen",
    "[NH]([!H])[!H]": "Secondary amine",
    "[O]=[!H]": "Carbonyl group",
    "[NH]1[!H][!H][!H]1": "Three-membered ring amine",
    "[OH][!H]": "Alcohol",
    "[NH]=[!H]": "Imine",
    "[O-]([!H])": "Oxygen anion",
    "[NH2][!H]": "Primary amine",
    "[o](:[!H]):[!H]": "Aromatic oxygen",
    "[N+]([!H])([!H])([!H])[!H]": "Quaternary ammonium",
    "[S]([!H])[!H]": "Thioether",
    "[N+]([!H])([!H])=[!H]": "Charged Secondary amine",
    "[S]=[!H]": "Thione",
    "[N]#C": "Isocyano group",
    "[S]([!H])([!H])=[!H]": "Sulfone",
    "[NH+]([R1])([R2])[R3]": "Charged Tertiary amine",
    "[S]([!H])([!H])(=[!H])=[!H]": "Sulfate",
    "[NH+]([!H])=[!H]": "Protonated imine",
    "[SH][!H]": "Thiol",
    "[NH2+][!H]": "Charged Primary amine",
    "[s](:[!H]):[!H]": "Aromatic sulfur",
    "[NH2+]=[!H]": "Charged imine",
    "[s](=[!H])(:[!H]):[!H]": "Aromatic sulfoxide",
    "[NH3+][!H]": "Ammonium ion",
    "[P]([!H])([!H])[!H]": "Phosphine",
    "[n](:[!H]):[!H]": "Aromatic nitrogen",
    "[P]([!H])=[!H]": "Phosphine oxide",
    "[n](:[!H])(:[!H]):[!H]": "Aromatic pyridine-like nitrogen",
    "[P]([!H])([!H])([!H])=[!H]": "Phosphonium salt",
    "[n]([!H]):[!H]:[!H]": "Aromatic nitrogen with single bond",
    "[PH]([!H])(=[!H])=[!H]": "Phosphine with double bond",
    "[n](=[!H])(:[!H]):[!H]": "Pyridine N-oxide"
}

# Run the test function to validate the SMARTS patterns
test_smarts_patterns(group_descriptions)

tpsa_data = [
    ("[N]([R1])([R2])[R3]", 3.24),  # Tertiary amine (single bonds to 3 non-H atoms)
    ("[nH](:[!H]):[!H]", 15.79),  # Aromatic nitrogen with hydrogen
    ("[N]([!H])=[!H]", 12.36),  # Secondary amine (with double bond)
    ("[n+](:[!H])(:[!H]):[!H]", 4.10),  # Aromatic charged nitrogen
    ("[N]#[!H]", 23.79),  # Nitrile group (triple bond)
    ("[n+]([!H]):[!H]:[!H]", 3.88),  # Aromatic charged nitrogen
    ("[N](=O)(=O)", 11.68),  # Nitro group (with double bonds)
    ("[nH+](:[!H]):[!H]", 14.14),  # Aromatic nitrogen with hydrogen and positive charge
    ("[N]#N", 13.60),  # Azide group (with double and triple bonds)
    ("[O]([!H])[!H]", 9.23),  # Ether (single bonds to two non-H atoms)
    ("[N]1([!H])[!H][!H][!H]1", 3.01),  # Three-membered ring nitrogen
    ("[O]1[!H][!H][!H]1", 12.53),  # Three-membered ring oxygen
    ("[NH]([!H])[!H]", 12.03),  # Primary amine with single bond
    ("[O]=[!H]", 17.07),  # Carbonyl group (double bond)
    ("[NH]1[!H][!H][!H]1", 21.94),  # Three-membered ring amine
    ("[OH][!H]", 20.23),  # Alcohol (single bond)
    ("[NH]=[!H]", 23.85),  # Imine (double bond)
    ("[O-]([!H])", 23.06),  # Oxygen anion (single bond)
    ("[NH2][!H]", 26.02),  # Primary amine (single bond)
    ("[o](:[!H]):[!H]", 13.14),  # Aromatic oxygen
    ("[N+]([!H])([!H])([!H])[!H]", 0.00),  # Quaternary ammonium (single bonds)
    ("[S]([!H])[!H]", 25.30),  # Thioether (single bonds)
    ("[N+]([!H])([!H])=[!H]", 3.01),  # Charged secondary amine (with double bond)
    ("[S]=[!H]", 32.09),  # Thione (double bond)
    ("[N]#C", 4.36),  # Isocyano group (triple bond)
    ("[S]([!H])([!H])=[!H]", 19.21),  # Sulfone (with double bond)
    ("[NH+]([R1])([R2])[R3]", 4.44),  # Charged tertiary amine
    ("[S]([!H])([!H])(=[!H])=[!H]", 8.38),  # Sulfate (with double bonds)
    ("[NH+]([!H])=[!H]", 13.97),  # Charged secondary amine (with double bond)
    ("[SH][!H]", 38.80),  # Thiol (single bond)
    ("[NH2+][!H]", 16.61),  # Charged primary amine (single bond)
    ("[s](:[!H]):[!H]", 28.24),  # Aromatic sulfur
    ("[NH2+]=[!H]", 25.59),  # Charged imine (double bond)
    ("[s](=[!H])(:[!H]):[!H]", 21.70),  # Aromatic sulfoxide (double bond)
    ("[NH3+][!H]", 27.64),  # Ammonium ion (single bond)
    ("[P]([!H])([!H])[!H]", 13.59),  # Phosphine (single bonds)
    ("[n](:[!H]):[!H]", 12.89),  # Aromatic nitrogen
    ("[P]([!H])=[!H]", 34.14),  # Phosphine oxide (double bond)
    ("[n](:[!H])(:[!H]):[!H]", 4.41),  # Aromatic pyridine-like nitrogen
    ("[P]([!H])([!H])([!H])=[!H]", 9.81),  # Phosphonium salt (double bond)
    ("[n]([!H]):[!H]:[!H]", 4.93),  # Aromatic nitrogen with single bond
    ("[PH]([!H])(=[!H])=[!H]", 23.47),  # Phosphine with double bond
    ("[n](=[!H])(:[!H]):[!H]", 8.39),  # Pyridine N-oxide (double bond)
]

descriptive_tpsa_data = [(group_descriptions[smarts], tpsa) for smarts, tpsa in tpsa_data]
tpsa_values_df = pd.DataFrame.from_records(descriptive_tpsa_data, columns=['group', 'tpsa_contribution'])
tpsa_values_df.to_csv(os.path.join('tpsa_saved_data', 'tpsa_values.csv'), index=False)

tpsa_smarts_df = pd.DataFrame.from_dict(group_descriptions, orient='index', columns=["Functional Group"])
tpsa_smarts_df.reset_index(inplace=True)
tpsa_smarts_df.columns = ["SMILES", "Functional Group"]
tpsa_smarts_df.to_csv(os.path.join('tpsa_saved_data', 'tpsa_smarts.csv'))
