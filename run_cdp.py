import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import warnings
import pickle
from torch.utils.data import DataLoader
from typing import List, Dict, Any
import torch
import argparse
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.data import InMemoryDataset
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.data import DataLoader
from torch_geometric.data import InMemoryDataset
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import MessagePassing, max_pool
from torch_geometric.nn import GCNConv
from mixture_of_experts import MoE
from torchvision.models import resnet18
from PIL import Image
import torchvision.transforms as T
from einops import rearrange
import time
from time import strftime, gmtime
import timm
import re
import optuna
warnings.filterwarnings("ignore")


# Argument parser
parser = argparse.ArgumentParser()
parser.add_argument("--num_subgraph_layers", type=int, default=1)
parser.add_argument("--num_experts", type=int, default=4)
parser.add_argument("--loss_coef", type=float, default=1.0e-2)
parser.add_argument("--out_length", type=int, default=82)
parser.add_argument("--in_channels", type=int, default=8)
parser.add_argument("--SEED", type=int, default=1234)
parser.add_argument("--show_every", type=int, default=10)
parser.add_argument("--val_every", type=int, default=5)
parser.add_argument("--small_dataset", action="store_true", default=False)
parser.add_argument("--end_epoch", type=int, default=0)
parser.add_argument("--prod_mode", type=str, default='trained_params')
parser.add_argument("--max_n_guesses", type=int, default=1)
parser.add_argument("--horizon", type=int, default=30)
parser.add_argument("--miss_threshold", type=float, default=2.0)
parser.add_argument("--funsion_type", type=str, default="None")
parser.add_argument("--img_encoder_type", type=str, default="None")
parser.add_argument("--data_dir", type=str, default='./data/crossing_90_a_04')
parser.add_argument("--debug", action="store_true", default=False)
parser.add_argument("--scheduler_type", type=str, default="None")
parser.add_argument("--lamda", type=float, default=1.0e-3)
parser.add_argument("--img_lr", type=float, default=2.0e-3)
parser.add_argument("--base_lr", type=float, default=1.0e-2)
parser.add_argument("--layer_width", type=int, default=256)
parser.add_argument("--num_fc_layers", type=int, default=1)
parser.add_argument("--alpha", type=float, default=0.5)
parser.add_argument("--device", type=str, default="cuda:0", help="gpu id")
parser.add_argument("--optuna", action="store_true", default=False)
parser.add_argument("--mask_gam", action="store_true", default=False)
parser.add_argument("--mask_carf", action="store_true", default=False)
parser.add_argument("--mask_moep", action="store_true", default=False)
parser.add_argument("--loss_type", type=str, default="both")
parser.add_argument("--print_shape", action="store_true", default=False)
args = parser.parse_args()
argsDict = args.__dict__


out_length = args.out_length
best_minade = float('inf')
np.random.seed(args.SEED)
torch.manual_seed(args.SEED)



time_str = strftime("%Y-%m-%d_%H-%M-%S", gmtime(time.time()))
results_dir = os.path.join(args.data_dir, time_str)
os.makedirs(results_dir, exist_ok=True)

if args.funsion_type == "None" and args.img_encoder_type == "None":
    log_dir = os.path.join(results_dir, 'GAT_Logs.txt')
else:
    log_dir = os.path.join(results_dir, f'GAT_{args.funsion_type}_{args.img_encoder_type}_num_fc_layers_{args.num_fc_layers}_img_lr_{args.img_lr}_lamda_{args.lamda}_Logs.txt')

for eachArg, value in argsDict.items():
    print(eachArg + ': ' + str(value))

with open(log_dir, 'a+') as f:
    f.writelines('------------------ start ------------------' + '\n')
    f.writelines(time_str + '\n')
    for eachArg, value in argsDict.items():
        f.writelines(eachArg + ': ' + str(value) + '\n')
    f.writelines('------------------- end -------------------' + '\n')


def get_fc_edge_index(num_nodes, start=0):
    """
    return a tensor(2, edges), indicing edge_index
    """
    to_ = np.arange(num_nodes, dtype=np.int64)
    edge_index = np.empty((2, 0))
    for i in range(num_nodes):
        from_ = np.ones(num_nodes, dtype=np.int64) * i
        # FIX BUG: no self loop in ful connected nodes graphs
        edge_index = np.hstack((edge_index, np.vstack((np.hstack([from_[:i], from_[i+1:]]), np.hstack([to_[:i], to_[i+1:]])))))
    edge_index = edge_index + start

    return edge_index.astype(np.int64), num_nodes + start

class GraphData(Data):
    """
    override key `cluster` indicating which polyline_id is for the vector
    """

    def __inc__(self, key, value, *args, **kwargs):

      if key == 'edge_index':
        return self.x.size(0)
      elif key == 'cluster':
        return int(self.cluster.max().item()) + 1
      else:
        return 0




class GraphDataset(InMemoryDataset):
    """
    dataset object similar to `torchvision`
    """

    def __init__(self, root, transform=None, pre_transform=None):
        super(GraphDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return ['dataset.pt']

    def download(self):
        pass

    def process(self):

        def get_data_path_ls(dir_):
            return [os.path.join(dir_, data_path) for data_path in os.listdir(dir_)]

        # make sure deterministic results
        data_path_ls = sorted(get_data_path_ls(self.root))

        valid_len_ls = []
        valid_len_ls = []
        data_ls = []
        for data_p in tqdm(data_path_ls):

          if not data_p.endswith('pkl'):
              continue
          x_ls = []
          y = None
          cluster = None
          start = None
          edge_index_ls = []

          data = pd.read_pickle(data_p)
          all_in_features = data['POLYLINE_FEATURES'].values[0]
          add_len = data['TARJ_LEN'].values[0]
          cluster = all_in_features[:, -1].reshape(-1).astype(np.int32)
          valid_len_ls.append(cluster.max())
          y = data['GT'].values[0].reshape(-1).astype(np.float32)
          start  =data['START'].values[0].reshape(-1).astype(np.float32)
          ID  =data['ID']
          # traj_mask, lane_mask = data["TRAJ_ID_TO_MASK"].values[0], data['LANE_ID_TO_MASK'].values[0]
          traj_mask = data["TRAJ_ID_TO_MASK"].values[0]
          agent_id = 0
          edge_index_start = 0
          # assert all_in_features[agent_id][-1] == 0, f"agent id is wrong. id {agent_id}: type {all_in_features[agent_id][4]}"

          for id_, mask_ in traj_mask.items():
              data_ = all_in_features[mask_[0]:mask_[1]]
              edge_index_, edge_index_start = get_fc_edge_index(
                  data_.shape[0], start=edge_index_start)
              x_ls.append(data_)
              edge_index_ls.append(edge_index_)
          edge_index = np.hstack(edge_index_ls)
          x = np.vstack(x_ls)
          data_ls.append([x, y, cluster, edge_index,start,ID])

        # [x, y, cluster, edge_index, valid_len]
        g_ls = []
        padd_to_index = np.max(valid_len_ls)
        feature_len = data_ls[0][0].shape[1]
        print(ID)
        for ind, tup in enumerate(data_ls):
            tup[0] = np.vstack(
                [tup[0], np.zeros((padd_to_index - tup[2].max(), feature_len), dtype=tup[0].dtype)])
            tup[2] = np.hstack(
                [tup[2], np.arange(tup[2].max()+1, padd_to_index+1)])
            g_data = GraphData(
                x=torch.from_numpy(tup[0]),
                y=torch.from_numpy(tup[1]),
                cluster=torch.from_numpy(tup[2]),
                edge_index=torch.from_numpy(tup[3]),
                valid_len=torch.tensor([valid_len_ls[ind]]),
                time_step_len=torch.tensor([padd_to_index + 1]),
                start=torch.from_numpy(tup[4]),
                ID=tup[5],
            )
            g_ls.append(g_data)
        data, slices = self.collate(g_ls)
        torch.save((data, slices), self.processed_paths[0])

class TrajPredMLP(nn.Module):

    def __init__(self, in_channels, out_channels, hidden_unit):
        super(TrajPredMLP, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, hidden_unit),
            nn.LayerNorm(hidden_unit),
            nn.ReLU(),
            nn.Linear(hidden_unit, out_channels),
        )

    def forward(self, x):
        return self.mlp(x)

def masked_softmax(X, valid_len):
    # Create a mask for invalid entries in each sequence
    mask = (torch.arange(X.shape[-1])[None, :].to(args.device) < valid_len[:, None]).to(args.device)

    # Expand mask dimensions to match X
    mask = mask.unsqueeze(1)

    # Apply the mask to X to set invalid entries to a very negative value
    X = X.masked_fill(~mask, -1e6)

    # Compute the softmax along the last dimension
    softmax_X = F.softmax(X, dim=-1)

    return softmax_X

class SelfAttentionLayer(nn.Module):

    def __init__(self, in_channels, global_graph_width, need_scale=False):
        super(SelfAttentionLayer, self).__init__()
        self.in_channels = in_channels
        self.q_lin = nn.Linear(in_channels, global_graph_width)
        self.k_lin = nn.Linear(in_channels, global_graph_width)
        self.v_lin = nn.Linear(in_channels, global_graph_width)
        self.scale_factor_d = 1 + \
            int(np.sqrt(self.in_channels)) if need_scale else 1

    def forward(self, x, valid_len):
        query = self.q_lin(x)
        key = self.k_lin(x)
        value = self.v_lin(x)
        scores = torch.bmm(query, key.transpose(1, 2))
        attention_weights = masked_softmax(scores, valid_len)
        return torch.bmm(attention_weights, value)

from torch_geometric.nn import GATConv

class SubGraph(nn.Module):

    def __init__(self, in_channels, num_subgraph_layres=3, hidden_unit=64):
        super(SubGraph, self).__init__()
        self.num_subgraph_layres = num_subgraph_layres
        self.hidden_unit = hidden_unit
        self.out_channels = hidden_unit

        self.layer_seq = nn.Sequential()
        for i in range(num_subgraph_layres):
            self.layer_seq.add_module(
                f'glp_{i}', GraphLayerProp(in_channels, hidden_unit))
            in_channels *= 2

    def forward(self, sub_data):
        x, edge_index = sub_data.x, sub_data.edge_index
        for name, layer in self.layer_seq.named_modules():
            if isinstance(layer, GraphLayerProp):
                x = layer(x, edge_index)
        sub_data.x = x
        out_data = max_pool(sub_data.cluster, sub_data)
        try:
          remainder = out_data.x.shape[0] % int(sub_data.time_step_len[0])
          if remainder != 0:
              new_size = out_data.x.shape[0] - remainder
              out_data.x = out_data.x[:new_size]  # 通过切片操作来调整大小
          assert out_data.x.shape[0] % int(sub_data.time_step_len[0]) == 0
        except:
            from pdb import set_trace; set_trace()
        out_data.x = out_data.x / out_data.x.norm(dim=0)
        return out_data



class GraphLayerProp(MessagePassing):

    def __init__(self, in_channels, hidden_unit=64, verbose=False):
        super(GraphLayerProp, self).__init__(node_dim=0)  # Initialize the base class
        self.verbose = verbose
        heads=  1
        self.heads = heads

        # Define the GAT layer
        self.gat_conv = GATConv(in_channels, hidden_unit, heads=heads, concat=True)
        # Linear layer to ensure the output dimension remains the same
        self.lin = nn.Linear(hidden_unit * heads, in_channels)

    def forward(self, x, edge_index):
        if self.verbose:
            print(f'x before GAT: {x}')

        # Apply GAT convolution
        x = self.gat_conv(x, edge_index)

        if self.verbose:
            print(f"x after GAT: {x}")

        # Apply the linear layer
        x = self.lin(x)

        if self.verbose:
            print(f"x after linear layer: {x}")
        return self.propagate(edge_index, size=(x.size(0), x.size(0)), x=x)

    def message(self, x_j, edge_index, size):
        # The message function is now handled internally by GATConv
        return x_j

    def update(self, aggr_out,x):
        # The update function can be omitted as it's not needed for GAT
        return torch.cat([x, aggr_out], dim=1)

class CrossAttention(nn.Module):
    def __init__(self, d, embed_dim=64, depth=1, nhead=1, dropout_pro=0.1):
        super(CrossAttention, self).__init__()
        self.attn = nn.ModuleList([
            nn.TransformerDecoderLayer(d_model=embed_dim, nhead=nhead, dropout=dropout_pro, batch_first=False)
            for _ in range(depth)
        ])

    def forward(self, coords, imgs):
        for layer in self.attn:
            coords = layer(coords, imgs)
        return coords

class ImageEncoder(nn.Module):
    def __init__(self,):
        super().__init__()
        self.resnet = resnet18(pretrained=True)
        self.resnet.avgpool = nn.Identity()
        self.resnet.fc = nn.Identity()
        self.mlp = nn.Linear(512, 64)
    
    def forward_feature(self, x):
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)
        
        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x)
        return x
    
    def forward(self, imgs):
        """
        imgs: T C H W
        """
        out: torch.Tensor = self.forward_feature(imgs)
        out = out.mean(dim=0, keepdim=True)
        out = out.reshape(*out.shape[:2], -1)
        out = self.mlp(out.permute(0, 2, 1))
        return out

class STTransformerEncoder(nn.Module):
    def __init__(self, ):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=64, 
                                       nhead=4, 
                                       dim_feedforward=256,
                                       batch_first=True)
        ])
        self.patch_embedding = nn.Conv2d(3, 64, 16, 16)
        self.pos_embedding = nn.Parameter(torch.zeros(1, 1, 64))
    
    def forward(self, imgs):
        x = rearrange(self.patch_embedding(imgs),
                           't c h w -> t (h w) c')
        x = x + self.pos_embedding.expand_as(x)
        for layer in self.layers:
            x = layer(x)
        return x.mean(1)

class HGNN(nn.Module):

    def __init__(self, in_channels, out_channels, num_subgraph_layers=1, num_global_graph_layer=1, subgraph_width=64, global_graph_width=64, traj_pred_mlp_width=64, num_experts=4, loss_coef=1.0e-2, img_encoder_type="None", lamda=1.0, num_fc_layers=1):
        super(HGNN, self).__init__()
        self.polyline_vec_shape = in_channels * (2 ** num_subgraph_layers)

        if args.mask_gam:
            global_graph_width = self.polyline_vec_shape

        self.subgraph = SubGraph(in_channels, num_subgraph_layers, subgraph_width)

        self.self_atten_layer = SelfAttentionLayer(self.polyline_vec_shape, global_graph_width, need_scale=False)

        self.traj_pred_mlp = TrajPredMLP(global_graph_width, out_channels, traj_pred_mlp_width)

        self.moe = MoE(
              dim = global_graph_width,
              num_experts = num_experts,               # increase the experts (# parameters) of your model without increasing computation
              hidden_dim = global_graph_width * 4,           # size of hidden dimension in each expert, defaults to 4 * dimension
              activation = nn.LeakyReLU,      # use your preferred activation, will default to GELU
              second_policy_train = 'random', # in top_2 gating, policy for whether to use a second-place expert
              second_policy_eval = 'random',  # all (always) | none (never) | threshold (if gate value > the given threshold) | random (if gate value > threshold * random_uniform(0, 1))
              second_threshold_train = 0.2,
              second_threshold_eval = 0.2,
              capacity_factor_train = 1.25,   # experts have fixed capacity per batch. we need some extra capacity in case gating is not perfectly balanced.
              capacity_factor_eval = 2.,      # capacity_factor_* should be set to a value >=1
              loss_coef = loss_coef                # multiplier on the auxiliary expert balancing auxiliary loss
          )
        self.lamda = lamda
        self.num_fc_layers = num_fc_layers
        if img_encoder_type == "R50" or img_encoder_type == "ViT_Small":
            if img_encoder_type == "R50":
                self.image_encoder = timm.create_model(
                    'resnet50',
                    pretrained=True,
                    num_classes=0,
                    pretrained_cfg_overlay=dict(file='./prtrained_models/resnet50_a1h_in1k/pytorch_model.bin')
                )
                if self.num_fc_layers == 3:
                    self.fc = nn.Sequential(
                        nn.Linear(in_features=2048, out_features=1024),
                        nn.ReLU(),
                        nn.Linear(in_features=1024, out_features=512),
                        nn.ReLU(),
                        nn.Linear(in_features=512, out_features=global_graph_width)
                    )
                elif self.num_fc_layers == 2:
                    self.fc = nn.Sequential(
                        nn.Linear(in_features=2048, out_features=512),
                        nn.ReLU(),
                        nn.Linear(in_features=512, out_features=global_graph_width)
                    )
                else:
                    self.fc = nn.Linear(in_features=2048, out_features=global_graph_width)
            elif img_encoder_type == "ViT_Small":
                self.image_encoder = timm.create_model(
                    'vit_small_patch16_224',
                    pretrained=True,
                    num_classes=0,
                    pretrained_cfg_overlay=dict(file='./prtrained_models/vit_small_patch16_224_augreg_in21k_ft_in1k/pytorch_model.bin', custom_load=False)
                )
                if self.num_fc_layers == 3:
                    self.fc = nn.Sequential(
                        nn.Linear(in_features=384, out_features=512),
                        nn.ReLU(),
                        nn.Linear(in_features=512, out_features=512),
                        nn.ReLU(),
                        nn.Linear(in_features=512, out_features=global_graph_width)
                    )
                elif self.num_fc_layers == 2:
                    self.fc = nn.Sequential(
                        nn.Linear(in_features=384, out_features=512),
                        nn.ReLU(),
                        nn.Linear(in_features=512, out_features=global_graph_width)
                    )
                else:
                    self.fc = nn.Linear(in_features=384, out_features=global_graph_width)
            self.cross_attention = CrossAttention(d=global_graph_width, embed_dim=global_graph_width, depth=1, nhead=1, dropout_pro=0.1)

    def forward(self, data, imgs, funsion_type="None"):
        time_step_len = int(data.time_step_len[0])
        valid_lens = data.valid_len
        sub_graph_out = self.subgraph(data)

        x = sub_graph_out.x.view(-1, time_step_len, self.polyline_vec_shape)

        # Hierarchical GNN
        if args.mask_gam:
            out = x
        else:
            out = self.self_atten_layer(x, valid_lens)

        # CARF
        imgs = self.fc(self.image_encoder(imgs).unsqueeze(1)) # T 1 C

        if args.mask_carf:
            out = (imgs[np.random.choice(imgs.shape[0], out.shape[0], replace=True)] + out) / 2
        else:
            if funsion_type=="cross_residual" or funsion_type=="cross":
                if funsion_type=="cross_residual":
                    out = out + self.lamda * self.cross_attention(out, imgs)
                elif funsion_type=="cross":
                    out = self.cross_attention(out, imgs)

        # MoEP
        if args.mask_moep:
            pred = out
        else:
            pred, aux_loss = self.moe(out)

        pred = self.traj_pred_mlp(pred)

        pred = pred.squeeze()


        return pred


def custom_loss(out, y, alpha=0.5):

    pre=out.view(-1, args.out_length//2, 2)
    true=y.view(-1, args.out_length//2, 2)

    distances = torch.norm(pre - true, dim=2)  # Calculate the Euclidean distance along the last dimension

    # Calculate ADE (Average Displacement Error) for each sample
    ADE_per_sample = torch.mean(distances, dim=1)  # Calculate mean distance along the second dimension (48)
    # Calculate the mean ADE and FDE across all samples
    mean_ADE = torch.mean(ADE_per_sample)

    FDE_per_sample = distances[:, -1]  # Take the last distance in each sample (at time step 48)

    mean_FDE = torch.mean(FDE_per_sample) #mean_ADE*alpha+(1-alpha)*mean_FDE

    if args.loss_type == "ade":
        return mean_ADE
    elif args.loss_type == "fde":
        return mean_FDE
    else:
        return mean_ADE*alpha+(1-alpha)*mean_FDE

def process_window_images(window: int, train_mode: bool = True):
    if train_mode:
        transforms = T.Compose([
            T.Resize((224, 224),),
            T.RandomRotation(90),
            T.ToTensor(),
            T.Normalize(mean=[0.2403, 0.2403, 0.2403], std=[0.2395, 0.2395, 0.2395]),
        ])
    else:
        transforms = T.Compose([
            T.Resize((224, 224),),
            T.ToTensor(),
            T.Normalize(mean=[0.2403, 0.2403, 0.2403], std=[0.2395, 0.2395, 0.2395]),
        ])
    f = lambda x: Image.open(os.path.join(args.data_dir, f'frames', f'gcf{x:04d}.jpg'))
    images = torch.stack([transforms(f(i)) for i in range(window, window + args.out_length)])
    images = images.to(args.device)
    return images

def start_train_window(window, funsion_type, img_encoder_type):
    epochs = 20000
    batch_size = 4096
    decay_lr_factor = 0.95
    decay_lr_every = 200
    imgs_train = process_window_images(window, train_mode=True)
    imgs_test = process_window_images(window, train_mode=False)

    TRAIN_DIR = os.path.join(args.data_dir, f'final', f'{window}window', 'intermediate')
    if funsion_type=="cross":
        checkpoint_dir = os.path.join(results_dir, f'{window}window', f'{window}window_moe_GAT_{img_encoder_type}_{funsion_type}.pth')
    elif funsion_type=="cross_residual":
        checkpoint_dir = os.path.join(results_dir, f'{window}window', f'{window}window_moe_GAT_{img_encoder_type}_{funsion_type}_lamda_{args.lamda}.pth')
    else:
        checkpoint_dir = os.path.join(results_dir, f'{window}window', f'{window}window_moe_GAT.pth')
    if not os.path.exists(os.path.dirname(checkpoint_dir)):
        os.makedirs(os.path.dirname(checkpoint_dir))

    train_data = GraphDataset(TRAIN_DIR)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=False)

    model = HGNN(
        args.in_channels,
        args.out_length,
        num_subgraph_layers=args.num_subgraph_layers,
        subgraph_width=args.layer_width,
        global_graph_width=args.layer_width,
        traj_pred_mlp_width=args.layer_width,
        num_experts=args.num_experts,
        loss_coef=args.loss_coef,
        img_encoder_type=img_encoder_type,
        lamda=args.lamda,
        num_fc_layers=args.num_fc_layers
    ).to(args.device)
    if img_encoder_type == "R50" or img_encoder_type == "ViT_Small":
        for name, param in model.image_encoder.named_parameters():
            param.requires_grad = False
        base_params =  [param for name, param in model.named_parameters() if not re.match(r'^(fc|image_encoder)', name)]
        optimizer = optim.Adam(
          [
            {"params": base_params},
            {"params": model.fc.parameters(), 'lr': args.img_lr},
          ], lr=args.base_lr
        )
    else:
        optimizer = optim.Adam(model.parameters(), lr=args.base_lr)
    if args.scheduler_type == "StepLR":
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=decay_lr_every, gamma=decay_lr_factor)
    else:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.95, patience=100, verbose=False, threshold=1e-4, threshold_mode='rel', cooldown=0, min_lr=0.001, eps=1e-08)

    global_step = 0
    model.train()
    best_loss = float('inf')  # Initialize with a large value
    best_model_state = None

    for epoch in range(epochs):
        num_samples = 1
        for data in train_loader:
            if epoch < args.end_epoch: break
            if isinstance(data, List):
                y = torch.cat([i.y for i in data], 0).view(-1, args.out_length).to(args.device)
            else:
                data = data.to(args.device)
                y = data.y.view(-1, args.out_length)
            y = data.y.view(-1, args.out_length).to(args.device)
            optimizer.zero_grad()
            out = model(data, imgs_train, funsion_type)
            loss = custom_loss(out, y, alpha=args.alpha)
            loss.backward()
            num_samples += y.shape[0]
            optimizer.step()
            global_step += 1
            if epoch % 500 == 0:
                with open(log_dir, 'a+') as f:
                    print( f"window {window}/epoch {epoch}/step {global_step}, loss:{loss.item():.6f}, GNN lr:{optimizer.state_dict()['param_groups'][0]['lr']:.6f}, Image lr:{optimizer.state_dict()['param_groups'][1]['lr']:.6f}")
                    print( f"window {window}/epoch {epoch}/step {global_step}, loss:{loss.item():.6f}, GNN lr:{optimizer.state_dict()['param_groups'][0]['lr']:.6f}, Image lr:{optimizer.state_dict()['param_groups'][1]['lr']:.6f}", file=f)
                    if args.debug:
                        for name, params in model.named_parameters():
                            print(name, params.grad.norm().item(), file=f)
        scheduler.step(loss)

        if loss < best_loss:
            best_loss = loss
            best_model_state = model.state_dict()
            best_optimizer=optimizer.state_dict()
            best_scheduler=scheduler.state_dict()

        if epoch % 500 == 0:
            if best_model_state is not None:
                torch.save({'epoch': epoch,
                    'model_state_dict': best_model_state,
                    'optimizer_state_dict': best_optimizer,
                    'loss': best_loss.item(),
                    'scheduler_state_dict': best_scheduler,}, checkpoint_dir)
                best_model = HGNN(
                    args.in_channels,
                    args.out_length,
                    num_subgraph_layers=args.num_subgraph_layers,
                    subgraph_width=args.layer_width,
                    global_graph_width=args.layer_width,
                    traj_pred_mlp_width=args.layer_width,
                    num_experts=args.num_experts,
                    loss_coef=args.loss_coef,
                    img_encoder_type=args.img_encoder_type,
                    lamda=args.lamda,
                    num_fc_layers=args.num_fc_layers
                ).to(args.device)
                best_model.load_state_dict(best_model_state)
    
    best_model.eval()
    with torch.no_grad():
        pre = None
        true = None
        for data in train_loader:
            if isinstance(data, List):
                y = torch.cat([i.y for i in data], 0).view(-1, args.out_length).to(args.device)
                start=torch.cat([i.start for i in data], 0).view(-1, 80).to(args.device)
            else:
                data = data.to(args.device)
                y = data.y.view(-1, args.out_length)
                start=data.start.view(-1, 80)
                ID  = data.ID

            pre = model(data, imgs_test, funsion_type)
            pre=pre.view(-1, args.out_length//2, 2)
            true=y.view(-1, args.out_length//2, 2)
            start=start.view(-1, 80//2, 2)
            ID  = data.ID
            # Calculate the Euclidean distance between each pair of predicted and true coordinates
            def calculate_metrics_at_time(pre, true, percentage):
                time_step = int((args.out_length//2 )* percentage / 100)
                if time_step == 0:
                    time_step = 1  # To handle 0% case, use the first time step

            # Select the data up to the specified time step
                pre_at_time = pre[:, :time_step, :]
                true_at_time = true[:, :time_step, :]

            # Calculate distances
                distances = torch.norm(pre_at_time - true_at_time, dim=2)

            # Calculate ADE
                ADE = torch.mean(distances, dim=1)  # Mean over the time dimension
                mean_ADE = torch.mean(ADE)  # Mean over the batch

            # Calculate FDE
                FDE = distances[:, -1]  # Last time step
                mean_FDE = torch.mean(FDE)  # Mean over the batch

                return mean_ADE.item(), mean_FDE.item()

            _25mean_ADE, _25mean_FDE = calculate_metrics_at_time(pre, true, 25)
            _50mean_ADE, _50mean_FDE = calculate_metrics_at_time(pre, true, 50)
            _75mean_ADE, _75mean_FDE = calculate_metrics_at_time(pre, true, 75)
            _100mean_ADE, _100mean_FDE = calculate_metrics_at_time(pre, true, 100)

    return _25mean_ADE, _25mean_FDE, _50mean_ADE, _50mean_FDE, _75mean_ADE, _75mean_FDE, _100mean_ADE, _100mean_FDE

def start_eval(model, train_loader, imgs, funsion_type):
    model.eval()
    with torch.no_grad():
        pre = None
        true = None
        for data in train_loader:
            if isinstance(data, List):
                y = torch.cat([i.y for i in data], 0).view(-1, args.args.out_length).to(args.device)
                start=torch.cat([i.start for i in data], 0).view(-1, 80).to(args.device)
            else:
                data = data.to(args.device)
                y = data.y.view(-1, args.out_lengthout_channels)
                start=data.start.view(-1, 80)
                ID  = data.ID

            pre = model(data, imgs, funsion_type)
            pre=pre.view(-1, args.out_length//2, 2)
            true=y.view(-1, args.out_length//2, 2)
            start=start.view(-1, 80//2, 2)
            ID  = data.ID
            # Calculate the Euclidean distance between each pair of predicted and true coordinates
            def calculate_metrics_at_time(pre, true, percentage):
                time_step = int((args.out_length//2 )* percentage / 100)
                if time_step == 0:
                    time_step = 1  # To handle 0% case, use the first time step

            # Select the data up to the specified time step
                pre_at_time = pre[:, :time_step, :]
                true_at_time = true[:, :time_step, :]

            # Calculate distances
                distances = torch.norm(pre_at_time - true_at_time, dim=2)

            # Calculate ADE
                ADE = torch.mean(distances, dim=1)  # Mean over the time dimension
                mean_ADE = torch.mean(ADE)  # Mean over the batch

            # Calculate FDE
                FDE = distances[:, -1]  # Last time step
                mean_FDE = torch.mean(FDE)  # Mean over the batch

                return mean_ADE.item(), mean_FDE.item()

            _25mean_ADE, _25mean_FDE = calculate_metrics_at_time(pre, true, 25)
            _50mean_ADE, _50mean_FDE = calculate_metrics_at_time(pre, true, 50)
            _75mean_ADE, _75mean_FDE = calculate_metrics_at_time(pre, true, 75)
            _100mean_ADE, _100mean_FDE = calculate_metrics_at_time(pre, true, 100)

    return _25mean_ADE, _25mean_FDE, _50mean_ADE, _50mean_FDE, _75mean_ADE, _75mean_FDE, _100mean_ADE, _100mean_FDE  


def objective(trial, keys, funsion_type, img_encoder_type):
    epochs = 20000
    batch_size = 4096
    decay_lr_factor = 0.95
    decay_lr_every = 200
    
    params = {
        "layer_width": trial.suggest_categorical("layer_width", [128, 256]),
        "img_lr": trial.suggest_categorical("img_lr", [1.0e-3, 2.0e-3, 5.0e-3]),
        "base_lr": trial.suggest_categorical("base_lr", [1.0e-2, 3.0e-2]),
        "lamda": trial.suggest_categorical("lamda", [1.0e-3, 2.0e-3, 5.0e-3]),
        "alpha": trial.suggest_categorical("alpha", [1.0e-1, 2.0e-1, 3.0e-1, 4.0e-1, 5.0e-1]),
    }

    mean_ADE={}
    mean_FDE={}
    for window in tqdm(keys):
        imgs_train = process_window_images(window, train_mode=True)
        TRAIN_DIR = os.path.join(args.data_dir, f'final', f'{window}window', 'intermediate')
        if funsion_type=="cross_residual" or funsion_type=="cross":
            checkpoint_dir = os.path.join(results_dir, f'{window}window', f'{window}window_moe_GAT_{img_encoder_type}_{funsion_type}.pth')
        else:
            checkpoint_dir = os.path.join(results_dir, f'{window}window', f'{window}window_moe_GAT.pth')
        if not os.path.exists(os.path.dirname(checkpoint_dir)):
            os.makedirs(os.path.dirname(checkpoint_dir))

        train_data = GraphDataset(TRAIN_DIR)

        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=False)

        model = HGNN(
            args.in_channels,
            args.out_length,
            num_subgraph_layers=args.num_subgraph_layers,
            subgraph_width=params["layer_width"],
            global_graph_width=params["layer_width"],
            traj_pred_mlp_width=params["layer_width"],
            num_experts=args.num_experts,
            loss_coef=args.loss_coef,
            img_encoder_type=img_encoder_type,
            lamda=params["lamda"],
            num_fc_layers=args.num_fc_layers
        ).to(args.device)

        if img_encoder_type == "R50" or img_encoder_type == "ViT_Small":
            for name, param in model.image_encoder.named_parameters():
                param.requires_grad = False
            base_params =  [param for name, param in model.named_parameters() if not re.match(r'^(fc|image_encoder)', name)]
            optimizer = optim.Adam(
                [
                    {"params": base_params},
                    {"params": model.fc.parameters(), 'lr': params["img_lr"]},
                ], lr=params["base_lr"]
            )
        else:
            optimizer = optim.Adam(model.parameters(), lr=params["base_lr"])

        if args.scheduler_type == "StepLR":
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=decay_lr_every, gamma=decay_lr_factor)
        else:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.95, patience=100, verbose=False, threshold=1e-4, threshold_mode='rel', cooldown=0, min_lr=0.001, eps=1e-08)

        global_step = 0
        model.train()
        best_loss = float('inf')  # Initialize with a large value
        best_model_state = None

        for epoch in range(epochs):
            num_samples = 1
            for data in train_loader:
                if epoch < args.end_epoch: break
                if isinstance(data, List):
                    y = torch.cat([i.y for i in data], 0).view(-1, args.out_length).to(args.device)
                else:
                    data = data.to(args.device)
                    y = data.y.view(-1, args.out_length)
                y = data.y.view(-1,args.out_length).to(args.device)
                optimizer.zero_grad()
                out = model(data, imgs_train, funsion_type)
                loss = custom_loss(out, y, alpha=params["alpha"])
                loss.backward()
                num_samples += y.shape[0]
                optimizer.step()
                global_step += 1
                if epoch % 500 == 0:
                    with open(log_dir, 'a+') as f:
                        print( f"window {window}/epoch {epoch}/step {global_step}, loss:{loss.item():.6f}, GNN lr:{optimizer.state_dict()['param_groups'][0]['lr']:.6f}, Image lr:{optimizer.state_dict()['param_groups'][1]['lr']:.6f}")
                        print( f"window {window}/epoch {epoch}/step {global_step}, loss:{loss.item():.6f}, GNN lr:{optimizer.state_dict()['param_groups'][0]['lr']:.6f}, Image lr:{optimizer.state_dict()['param_groups'][1]['lr']:.6f}", file=f)
                        if args.debug:
                            for name, params in model.named_parameters():
                                print(name, params.grad.norm().item(), file=f)
            scheduler.step(loss)

            if loss < best_loss:
                best_loss = loss
                best_model_state = model.state_dict()
                best_optimizer=optimizer.state_dict()
                best_scheduler=scheduler.state_dict()

            if epoch % 500 == 0:
                if best_model_state is not None:
                    torch.save({'epoch': epoch,
                        'model_state_dict': best_model_state,
                        'optimizer_state_dict': best_optimizer,
                        'loss': best_loss.item(),
                        'scheduler_state_dict': best_scheduler,}, checkpoint_dir)
                    best_model = HGNN(
                        args.in_channels,
                        args.out_length,
                        num_subgraph_layers=args.num_subgraph_layers,
                        subgraph_width=params["layer_width"],
                        global_graph_width=params["layer_width"],
                        traj_pred_mlp_width=params["layer_width"],
                        num_experts=args.num_experts,
                        loss_coef=args.loss_coef,
                        img_encoder_type=img_encoder_type,
                        lamda=params["lamda"],
                        num_fc_layers=args.num_fc_layers
                        ).to(args.device)
                    best_model.load_state_dict(best_model_state)
        
        imgs_test = process_window_images(window, train_mode=False)
        best_model.eval()
        with torch.no_grad():
            pre = None
            true = None
            for data in train_loader:
                if isinstance(data, List):
                    y = torch.cat([i.y for i in data], 0).view(-1, args.out_length).to(args.device)
                    start=torch.cat([i.start for i in data], 0).view(-1, 80).to(args.device)
                else:
                    data = data.to(args.device)
                    y = data.y.view(-1, args.out_length)
                    start=data.start.view(-1, 80)
                    ID  = data.ID

                pre = model(data, imgs_test, funsion_type)
                pre=pre.view(-1, args.out_length//2, 2)
                true=y.view(-1, args.out_length//2, 2)
                start=start.view(-1, 80//2, 2)
                ID  = data.ID
                # Calculate the Euclidean distance between each pair of predicted and true coordinates
                distances = torch.norm(pre - true, dim=2)  # Calculate the Euclidean distance along the last dimension

                # Calculate ADE (Average Displacement Error) for each sample
                ADE_per_sample = torch.mean(distances, dim=1)  # Calculate mean distance along the second dimension (48)

                # Calculate FDE (Final Displacement Error) for each sample
                FDE_per_sample = distances[:, -1]  # Take the last distance in each sample (at time step 48)

                # Calculate the mean ADE and FDE across all samples
                mean_ADE[window] = torch.mean(ADE_per_sample).item()
                mean_FDE[window] = torch.mean(FDE_per_sample).item()

                with open(log_dir, 'a+') as f:
                    print(f"window: {window}, 100% ADE: {mean_ADE[window]:.4g}, 100% FDE: {mean_FDE[window]:.4g}")
                    print(f"window: {window}, 100% ADE: {mean_ADE[window]:.4g}, 100% FDE: {mean_FDE[window]:.4g}", file=f)

    ADE_values = [value for value in mean_ADE.values()]
    ADE_mean = np.mean(np.stack(ADE_values))

    FDE_values = [value for value in mean_FDE.values()]
    FDE_mean = np.mean(np.stack(FDE_values))

    print(f"Trial {trial.number}, ADE: {ADE_mean:.4f}, FDE: {FDE_mean:.4f}, Params: {params}")
    print(f"Trial {trial.number}, ADE: {ADE_mean:.4f}, FDE: {FDE_mean:.4f}, Params: {params}", file=open(log_dir, 'a+'))

    return ADE_mean, FDE_mean


if args.optuna:
    grid_keys = [0, 162, 243, 405, 1782]
    study = optuna.create_study(directions=["minimize", "minimize"], study_name="gcf", sampler=optuna.samplers.NSGAIISampler())
    func = lambda trial: objective(trial, grid_keys, args.funsion_type, args.img_encoder_type)
    study.optimize(func, n_trials=90)
    sorted_trials = sorted(
        [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE], key=lambda x: (x.values[1], x.values[0])  # 先按FDE排序，再按ADE排序
    )
    with open(log_dir, "a+") as f:
        for i, trial in enumerate(sorted_trials):
            f.writelines('------------------ start ------------------' + '\n')
            f.write(f"Trial {trial.number}, ADE: {trial.values[0]:.4f}, FDE: {trial.values[1]:.4f}\n")
            f.write("Parameters:\n")
            for k, v in trial.params.items():
                f.write(f"\t{k}: {v}\n")
            f.writelines('------------------ end ------------------' + '\n')
else:
    import gc
    gc.collect()
    _25mean_ADE={}
    _25mean_FDE={}
    _50mean_ADE={}
    _50mean_FDE={}
    _75mean_ADE={}
    _75mean_FDE={}
    _100mean_ADE={}
    _100mean_FDE={}
    keys = [0, 81, 162, 243, 324, 405, 486, 567, 648, 729, 810, 891, 972, 1053, 1134, 1215, 1296, 1377, 1458, 1539, 1620, 1701, 1782, 1863, 1944, 2025, 2106, 2187, 2268, 2349, 2430, 2511, 2592]
    uu = {}


    for i in tqdm(keys):
        try:
            _25mean_ADE[i], _25mean_FDE[i], _50mean_ADE[i], _50mean_FDE[i], _75mean_ADE[i], _75mean_FDE[i], _100mean_ADE[i], _100mean_FDE[i] = start_train_window(i, args.funsion_type, args.img_encoder_type)
            with open(log_dir, 'a+') as f:
                print(f"window: {i}, 25% ADE: {_25mean_ADE[i]:.4f}, 25% FDE: {_25mean_FDE[i]:.4f}, 50% ADE: {_50mean_ADE[i]:.4f}, 50% FDE: {_50mean_FDE[i]:.4f}, 75% ADE: {_75mean_ADE[i]:.4f}, 75% FDE: {_75mean_FDE[i]:.4f}, 100% ADE: {_100mean_ADE[i]:.4g}, 100% FDE: {_100mean_FDE[i]:.4f}")
                print(f"window: {i}, 25% ADE: {_25mean_ADE[i]:.4f}, 25% FDE: {_25mean_FDE[i]:.4f}, 50% ADE: {_50mean_ADE[i]:.4f}, 50% FDE: {_50mean_FDE[i]:.4f}, 75% ADE: {_75mean_ADE[i]:.4f}, 75% FDE: {_75mean_FDE[i]:.4f}, 100% ADE: {_100mean_ADE[i]:.4g}, 100% FDE: {_100mean_FDE[i]:.4f}", file=f)
        except:
            uu[i]=i

    tensor_values = [value for value in _25mean_ADE.values()]
    mean_value = np.mean(np.stack(tensor_values))
    with open(log_dir, 'a+') as f:
        print(f"ADE@25%: {mean_value:.4f}")
        print(f"ADE@25%: {mean_value:.4f}", file=f)


    tensor_values = [value for value in _50mean_ADE.values()]
    mean_value = np.mean(np.stack(tensor_values))
    with open(log_dir, 'a+') as f:
        print(f"ADE@50%: {mean_value:.4f}")
        print(f"ADE@50%: {mean_value:.4f}", file=f)


    tensor_values = [value for value in _75mean_ADE.values()]
    mean_value = np.mean(np.stack(tensor_values))
    with open(log_dir, 'a+') as f:
        print(f"ADE@75%: {mean_value:.4f}")
        print(f"ADE@75%: {mean_value:.4f}", file=f)


    tensor_values = [value for value in _100mean_ADE.values()]
    mean_value = np.mean(np.stack(tensor_values))
    with open(log_dir, 'a+') as f:
        print(f"ADE: {mean_value:.4f}")
        print(f"ADE: {mean_value:.4f}", file=f)


    tensor_values = [value for value in _100mean_FDE.values()]
    mean_value = np.mean(np.stack(tensor_values))
    with open(log_dir, 'a+') as f:
        print(f"FDE: {mean_value:.4f}")
        print(f"FDE: {mean_value:.4f}", file=f)
