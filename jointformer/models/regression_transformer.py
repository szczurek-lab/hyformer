import pandas as pd 


def _load_regression_transformer_dataset(filename):
    df = pd.read_csv(filename, sep='|', header=None)
    df.columns = ['property', 'smiles'] 
    df['property'] = df['property'].str.replace('<qed>', '', regex=False)
    df['property'] = df['property'].astype(float)
    dataset = []
    for _, row in df.iterrows():
        property = row['property']
        smiles = row['smiles']
        dataset.append((property, smiles))
    return dataset
