import pandas as pd

df = pd.read_csv(r'C:\Users\nvanwieren\Downloads\open_job_ops 3-17-2026.csv')

ignore_df = pd.read_excel(r'C:\Users\nvanwieren\Downloads\Ignore_Parts.xlsx')
ignore_parts = set(ignore_df['Ignore_Parts'].dropna())
df = df[~df['partNumber'].isin(ignore_parts)]

ops_to_drop = [
    'OS Assemble FG', 'Receive (Ea)', 'OS Pem FG', 'OS Form FG',
    'OS Weld FG', 'OS Machine FG', 'OS Laser FG', 'OS Grind FG',
]
df = df[~df['operationCode'].isin(ops_to_drop)]

part = df[df['partNumber'] == '122402'].copy()
print('All rows for 122402:')
print(part[['partNumber','operationCode','quantity','quantityCompleted','rate','runTime','setupTime','dueDate','jobOperationStatus']].to_string(index=True))

OP_MAP = {
    'Laser WIP':'Laser','Laser FG':'Laser',
    'Form WIP 1':'Press_Brake','Form FG':'Press_Brake',
    'Pem WIP 1':'PEM','Pem FG':'PEM','Pem WIP 2':'PEM',
    'OS Paint WIP':'Powder_Coat','OS Paint FG':'Powder_Coat',
    'Assemble WIP 1':'Assembly','Secondary Ops FG':'Assembly',
    'Secondary Ops WIP':'Assembly','Assemble FG STD':'Assembly',
    'Assemble FG SMP':'Assembly','Countersink WIP':'Assembly',
    'Tap FG':'Assembly','Cut to Length':'Assembly',
    'Weld FG':'Weld','Weld WIP 1':'Weld',
    'Resistance Weld WIP 1':'Resistance_Weld',
    'Pack FG SMP':'Packing','Pick & Pack FG':'Packing',
}

assy = part[part['operationCode'].isin([k for k,v in OP_MAP.items() if v == 'Assembly'])].copy()
assy['qty_remaining'] = (assy['quantity'].fillna(0) - assy['quantityCompleted'].fillna(0)).clip(lower=0)
assy['hours_calc'] = assy.apply(lambda r: r['qty_remaining'] / r['rate'] if r['rate'] > 0 else 0, axis=1)

print('\nAssembly ops only:')
print(assy[['operationCode','quantity','quantityCompleted','qty_remaining','rate','runTime','hours_calc']].to_string(index=True))
print(f'\nTotal assembly hours (qty_remaining / rate): {assy["hours_calc"].sum():.4f}')
print(f'Total assembly hours (runTime column):       {assy["runTime"].sum():.4f}')
