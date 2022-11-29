import streamlit as st
import torch
from torch_geometric.loader import DataLoader
from ogb.graphproppred import PygGraphPropPredDataset, Evaluator
from sklearn.metrics import roc_auc_score, average_precision_score
import pandas as pd
import numpy as np
from tqdm import tqdm
# pip install rdkit-pypi
from rdkit import Chem
from rdkit import RDLogger
from rdkit.Chem.Draw import IPythonConsole
from rdkit.Chem.Draw import MolsToGridImage

device = torch.device("cuda:" + str(0)) if torch.cuda.is_available() else torch.device("cpu")

st.title("가짜연구소 5기 그래프로 설득하기")

y_true = []
y_pred = []
ap_list = []
trt_labeled = []

dataset = PygGraphPropPredDataset(name = 'ogbg-molpcba')
split_idx = dataset.get_idx_split()
test_loader = DataLoader(dataset[split_idx["test"]], batch_size = 32, shuffle=False, num_workers = 0)
test_tensor = split_idx["test"]
test_numpy = test_tensor.numpy()

mol = pd.read_csv('./data/mol.csv')
af = pd.DataFrame(mol)

st.markdown("### Smiles list")
st.dataframe(af.smiles)

PATH = "./model/gin.pt"
model = torch.load(PATH)
model.eval()

def molecule_from_smiles(smiles):
    molecule = Chem.MolFromSmiles(smiles, sanitize=False)
    flag = Chem.SanitizeMol(molecule, catchErrors=True)
    if flag != Chem.SanitizeFlags.SANITIZE_NONE:
        Chem.SanitizeMol(molecule, sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL ^ flag)

    Chem.AssignStereochemistry(molecule, cleanIt=True, force=True)
    return molecule

for step, batch in enumerate(tqdm(test_loader, desc="Iteration")):
    batch = batch.to(device)

    if batch.x.shape[0] == 1:
        pass
    else:
        with torch.no_grad():
            pred = model(batch)

        y_true.append(batch.y.view(pred.shape).detach().cpu())
        y_pred.append(pred.detach().cpu())

y_true = torch.cat(y_true, dim = 0).numpy()
y_pred = torch.cat(y_pred, dim = 0).numpy()

for i in range(y_true.shape[1]):
    if np.sum(y_true[:,i] == 1) > 0 and np.sum(y_true[:,i] == 0) > 0:
        is_labeled = y_true[:,i] == y_true[:,i]
        trt_labeled.append(is_labeled[0])
        ap = average_precision_score(y_true[is_labeled,i], y_pred[is_labeled,i])

        ap_list.append(ap)
        
values = st.text_input('Select a range of values:')

molecules = [molecule_from_smiles(af.smiles[index]) for index in test_numpy]

st.write(MolsToGridImage(molecules, molsPerRow=10))

st.write(af.smiles[int(values)])

st.text_input("Your namasdfsadfsadfe", key="name")

st.session_state.name

