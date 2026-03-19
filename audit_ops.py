import pandas as pd

df = pd.read_csv(r'C:\Users\nvanwieren\Downloads\open_job_ops 3-17-2026.csv')

# Load ignore parts
ignore_df = pd.read_excel(r'C:\Users\nvanwieren\Downloads\Ignore_Parts.xlsx')
ignore_parts = ignore_df['Ignore_Parts'].dropna().tolist()
df = df[~df['partNumber'].isin(ignore_parts)]

ops_to_drop = ['OS Assemble FG', 'Receive (Ea)', 'OS Pem FG', 'OS Form FG', 'OS Weld FG', 'OS Machine FG']
df = df[~df['operationCode'].isin(ops_to_drop)]
print(f'Rows after filters: {len(df)}')

known_ops = [
    'Laser WIP', 'Form WIP 1', 'Pem WIP 1', 'OS Paint WIP', 'Assemble WIP 1',
    'Form FG', 'Weld FG', 'Weld WIP 1', 'OS Paint FG', 'Secondary Ops FG',
    'Pem FG', 'Pem WIP 2', 'Secondary Ops WIP', 'Resistance Weld WIP 1',
    'Assemble FG STD', 'Countersink WIP', 'Laser FG', 'Pack FG SMP', 'Tap FG',
    'Assemble FG SMP', 'Cut to Length', 'Pick & Pack FG'
]

unmapped = df[~df['operationCode'].isin(known_ops)]['operationCode'].value_counts()
print(f'\nUnmapped operationCodes ({len(unmapped)} unique):')
print(unmapped.to_string())

# Check for status/quantity mismatch: status not complete but qty_completed == qty
mismatch = df[
    (df['quantityCompleted'] >= df['quantity']) &
    (df['jobOperationStatus'].str.lower() != 'complete')
]
print(f'\nRows where quantityCompleted >= quantity but status is not complete: {len(mismatch)}')
if len(mismatch) > 0:
    print(mismatch[['partNumber', 'operationCode', 'quantity', 'quantityCompleted', 'jobOperationStatus']].head(10).to_string())
