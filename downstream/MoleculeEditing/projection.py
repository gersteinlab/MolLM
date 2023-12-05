#!/usr/bin/env python
# coding: utf-8

# In[2]:


import torch

all_mols = torch.load('embedding_data/all.pt', map_location='cuda')


# In[3]:


type(all_mols), all_mols[0].shape


# In[4]:


import os
import time
import argparse
from distutils.util import strtobool
from mflow.models.hyperparams import Hyperparameters
from mflow.utils.model_utils import load_model, get_latent_vec

parser = argparse.ArgumentParser()
parser.add_argument("--input_text", type=str, default='This molecule is beautiful.')
parser.add_argument("--checkpoint_name", type=str, default='littlegin=graphclinit_bert=scibert_epoch=299-step=18300.ckpt')
parser.add_argument("--model_dir", type=str, default='./results')
parser.add_argument("--data_dir", type=str, default='../data')
parser.add_argument('--data_name', type=str, default='qm9', choices=['qm9', 'zinc250k'], help='dataset name')
# parser.add_argument('--molecule_file', type=str, default='qm9_relgcn_kekulized_ggnp.npz',
#                     help='path to molecule dataset')
parser.add_argument("--snapshot-path", "-snapshot", type=str, required=True)
parser.add_argument("--hyperparams-path", type=str, default='moflow-params.json', required=True)
parser.add_argument("--gpu", type=int, default=-1)
parser.add_argument("--batch-size", type=int, default=100)
parser.add_argument('--additive_transformations', type=strtobool, default='false',
                    help='apply only additive coupling layers')
parser.add_argument('--delta', type=float, default=0.1)
parser.add_argument('--n_experiments', type=int, default=1, help='number of times generation to be run')
parser.add_argument('--seed', type=int, default=123)
parser.add_argument('--debug', action='store_true', default=False)
parser.add_argument('--temperature', type=float, default=1.0,
                    help='temperature of the gaussian distribution')
# parser.add_argument('--draw_neighborhood', type=strtobool, default='true',
#                     help='if neighborhood of a molecule to be visualized')

parser.add_argument('--save_fig', type=strtobool, default='true')
parser.add_argument('--save_score', type=strtobool, default='true')
parser.add_argument('-r', '--reconstruct', action='store_true', default=False)
# parser.add_argument('-i', '--interpolation', action='store_true', default=False)
parser.add_argument('--int2point', action='store_true', default=False)
parser.add_argument('--intgrid', action='store_true', default=False)

parser.add_argument('--inter_times', type=int, default=5)

parser.add_argument('--correct_validity', type=strtobool, default='true',
                    help='if apply validity correction after the generation')
args = parser.parse_args("--model_dir results/zinc250k_512t2cnn_256gnn_512-64lin_10flow_19fold_convlu2_38af-1-1mask -snapshot model_snapshot_epoch_200 --gpu 0 --data_name zinc250k --hyperparams-path moflow-params.json --temperature 0.85 --batch-size 1 --n_experiments 5".split(" "))

start = time.time()
print("Start at Time: {}".format(time.ctime()))
# chainer.config.train = False
snapshot_path = os.path.join(args.model_dir, args.snapshot_path)
hyperparams_path = os.path.join(args.model_dir, args.hyperparams_path)
print("loading hyperparamaters from {}".format(hyperparams_path))
model_params = Hyperparameters(path=hyperparams_path)
model = load_model(snapshot_path, model_params, debug=True)
if len(model.ln_var) == 1:
    print('model.ln_var: {:.2f}'.format(model.ln_var.item()))
elif len(model.ln_var) == 2:
    print('model.ln_var[0]: {:.2f}, model.ln_var[1]: {:.2f}'.format(model.ln_var[0].item(), model.ln_var[1].item()))

if args.gpu >= 0:
    # device = args.gpu
    device = torch.device('cuda:'+str(args.gpu) if torch.cuda.is_available() else 'cpu')
else:
    device = torch.device('cpu')
model.to(device)
print(f'device = {device}')
model.eval()  # Set model for evaluation


# In[5]:


# z = get_latent_vec(model, "O=C(NC[C@H]1CCCO1)c1ccccc1N1CCCC1=O", data_name='zinc250k')
from mflow.utils.model_utils import smiles_to_adj, rescale_adj

def smiles_to_moflow_rep(smiles):
    adj, atoms = smiles_to_adj(smiles, 'zinc250k')
    adj_normalized = rescale_adj(adj)
    with torch.no_grad():
        device = next(model.parameters()).device
        z0, _ = model(adj.to(device), atoms.to(device), adj_normalized.to(device))

    h, adj_h = z0
    # Flatten h and adj_h into 1D tensors
    h_flat = h.view(h.shape[0], -1)
    adj_h_flat = adj_h.view(adj_h.shape[0], -1)

    return torch.cat([h_flat, adj_h_flat], dim=1)


# In[6]:


with open('MoleculeSTM_editing_SMILES.txt', 'r') as file:
    smiles = file.readlines()


# In[7]:


from tqdm import tqdm

smiles_moflow_reps = []
for smile in tqdm(smiles):
    smiles_moflow_reps.append(smiles_to_moflow_rep(smile))


# In[8]:


import pickle

pickle.dump(smiles_to_moflow_rep, open('moflow_reps.pkl', 'wb'))


# In[9]:


# FROM MOFLOW REPRESENTATION TO MOLM REPRESENTATION

to_embeddings = [mol.squeeze(0).float() for mol in all_mols]
from_embeddings = [mol.squeeze(0).float() for mol in smiles_moflow_reps]

# from_embeddings[0].shape, to_embeddings[0].shape


# In[10]:


import torch
from torch import nn, optim

class ProjectionModel(nn.Module):
    def __init__(self):
        super(ProjectionModel, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(6156, 4096), # MO FLOW EMBEDDING = (6156)
            nn.ReLU(),
            nn.Linear(4096, 2048),
            nn.ReLU(),
            nn.Linear(2048, 768) # MOLM EMBEDDING (TRANSFORMER-M) = (756)
        )

    def forward(self, x):
        return self.model(x)

project_model = ProjectionModel()

# Define a loss function
# def custom_loss(y_pred, y_true):
#     return

# Define an optimizer
optimizer = optim.Adam(project_model.parameters(), lr=0.001)

device = 'cpu'

# Move model to the device
project_model = project_model.to(device)


# In[ ]:


from tqdm import tqdm

# Training loop
def train(model, from_embeddings, to_embeddings, optimizer, epochs):
    global device
    model.train()

    pbar_total = tqdm(total=epochs, desc="Training", ncols=70)

    try:
        for epoch in range(epochs):
            total_loss = 0
            pbar_epoch = tqdm(total=len(from_embeddings), desc=f"Epoch {epoch+1}", ncols=70, leave=False)

            for from_embed, to_embed in zip(from_embeddings, to_embeddings):
                torch.cuda.empty_cache()
                from_embed = from_embed.to(device)
                to_embed = to_embed.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward pass
                outputs = model(from_embed)

                # calculate loss
                loss = (outputs - to_embed).pow(2).sum(-1).mean()
                # print(f'outputs dev, to_embed dev, loss dev: {outputs.device}, {to_embed.device}, {loss.device}')

                # backward pass
                loss.backward()

                # update model parameters
                optimizer.step()

                total_loss += loss.item()
                pbar_epoch.update(1)

            pbar_epoch.close()

            avg_loss = total_loss / len(from_embeddings)
            print(f"Epoch {epoch+1}, Loss: {avg_loss}")

            pbar_total.update(1)
    except KeyboardInterrupt:
        print("\nTraining was interrupted. Saving the model...")

    finally:
        # Save model even if KeyboardInterrupt occurs
        torch.save(model.state_dict(), 'projection_model.bin')
        pbar_total.close()

# Train the model
train(project_model, from_embeddings, to_embeddings, optimizer, epochs=999)

# In[ ]:





# In[ ]:




