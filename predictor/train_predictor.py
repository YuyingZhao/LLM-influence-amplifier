"""Train the pairwise influence predictor for the weibo dataset."""

import os
import pickle
import random
import time
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as BaseDataset


def seed_everything(seed):
    """Seed RNGs for reproducibility.

    Args:
        seed: Integer seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def batch_to_gpu(batch, device):
    """Move batch tensors to the target device.

    Args:
        batch: Dict of tensors.
        device: Target torch.device.

    Returns:
        Updated batch dict on device.
    """
    for c in batch:
        batch[c] = batch[c].to(device)
    return batch

class Dataset(BaseDataset):
    """Dataset wrapper for pairwise influence instances."""
    def __init__(self, users, content, label, creator):
        """Initialize dataset fields.

        Args:
            users: Target user IDs.
            content: Content/post IDs.
            label: Influence labels.
            creator: Creator user IDs.
        """
        self.users = users
        self.content = content
        self.label = label
        self.creator = creator

    def _get_feed_dict(self, index):
        """Build a single training instance dict."""
        feed_dict = {
            'users': self.users[index],
            'content': self.content[index],
            'label': self.label[index],
            'creator': self.creator[index]
        }
        return feed_dict

    def __len__(self):
        """Return dataset size."""
        return len(self.users)

    def __getitem__(self, index):
        """Return a single instance by index."""
        return self._get_feed_dict(index)

    def collate_batch(self, feed_dicts):
        """Collate a batch of instances into tensors."""
        feed_dict = dict()
        feed_dict['users'] = torch.LongTensor([d['users'] for d in feed_dicts])
        feed_dict['creator'] = torch.LongTensor([d['creator'] for d in feed_dicts])
        feed_dict['content'] = torch.LongTensor([d['content'] for d in feed_dicts])
        feed_dict['label'] = torch.FloatTensor([d['label'] for d in feed_dicts]) # treat it as a regression problem
        return feed_dict

def pickle_load(file_name):
    """Load a pickle file.

    Args:
        file_name: Path to the pickle file.

    Returns:
        Deserialized object.
    """
    with open(file_name, 'rb') as f:
        return pickle.load(f)

def prepare_instances(pids, post_author_dict, reverse_repost_dict, two_hop_adj):
    """Create labeled influence instances from post IDs.

    Args:
        pids: List of post IDs.
        post_author_dict: Map of post to author.
        reverse_repost_dict: Map of post to reposting users.
        two_hop_adj: Two-hop adjacency for candidate neighbors.

    Returns:
        List of [user, creator, content, label] instances.
    """
    instances = []
    pos_cnt = 0
    neg_cnt = 0
    for post in pids:
        author = post_author_dict[post]
        neighbors = []
        if author in two_hop_adj:
            neighbors = two_hop_adj[author]
        repost_users = []
        if post in reverse_repost_dict:
            repost_users = reverse_repost_dict[post]
        pos_user = repost_users
        neg_user = set(neighbors).difference(repost_users)
        # [user, creator, content, label]
        for p in pos_user:
            instances.append([p, author, post, 1])
        neg_user = random.sample(list(neg_user), min(len(neg_user), len(pos_user)*2))
        for n in neg_user:
            instances.append([n, author, post, 0])
        pos_cnt += len(pos_user)
        neg_cnt += len(neg_user)
    print("Positive instances:", pos_cnt, "Negative instances:", neg_cnt)
    print("Total instances:", len(instances), 'Ratio:', pos_cnt/neg_cnt)
    return instances

class Prob_Model(nn.Module):
    """Pairwise influence prediction model."""
    def __init__(self, dataset_name):
        """Initialize the model and precompute embeddings.

        Args:
            dataset_name: Dataset name used to load preprocessed data.
        """
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        post_content_dict, post_author_dict, influence_dict = pickle_load('../datasets/{}/preprocessed/post_info.pkl'.format(dataset_name))
        follow_edges, two_hop_adj = pickle_load('../datasets/{}/preprocessed/network_info.pkl'.format(dataset_name))
        post_num = len(post_author_dict)
        users = []
        for u, v in follow_edges:
            users.append(u)
            users.append(v)
        user_num = max(users) + 1
        print("User num:", user_num, "Post num:", post_num)

        initializer = nn.init.xavier_uniform_
        self.user_embedding = nn.Parameter(initializer(torch.empty(user_num, 32)))
        
        if os.path.exists('./post_embeddings/{}.pt'.format(dataset_name)):
            self.post_embedding = torch.load('./post_embeddings/{}.pt'.format(dataset_name), weights_only=True)
            print("Finish loading post embeddings")
        else:
            # obtain post embeddings from sentence transformer
            self.sentence_model = SentenceTransformer('sentence-transformers/distiluse-base-multilingual-cased-v2')
            self.sentence_model.eval()
            self.sentence_model.to(self.device)
            for parameters in self.sentence_model.parameters():
                parameters.requires_grad = False

            posts = []
            for i in range(post_num):
                post_content = post_content_dict[i]
                posts.append(post_content)
            post_embeddings = self.sentence_model.encode(posts)
            self.post_embedding = torch.from_numpy(post_embeddings).to(self.device)
            if not os.path.exists('./post_embeddings'):
                os.makedirs('./post_embeddings')
            torch.save(self.post_embedding, './post_embeddings/{}.pt'.format(dataset_name))
            print("Finish calculating post embeddings")

        layers = []
        in_size = 512
        for hidden_size in [128]:
            layers.append(nn.Linear(in_size, hidden_size))
            layers.append(nn.Dropout(0.5))
            layers.append(nn.ReLU())
            in_size = hidden_size
        layers.append(nn.Linear(128, 32))
        layers.append(nn.Dropout(0.5))
        self.post_dimension_reduction = nn.Sequential(*layers).to(self.device)

        layers = []
        in_size = 32
        for hidden_size in [64]:
            layers.append(nn.Linear(in_size, hidden_size))
            layers.append(nn.Dropout(0.5))
            layers.append(nn.ReLU())
            in_size = hidden_size
        layers.append(nn.Linear(64, 32))
        layers.append(nn.Dropout(0.5))
        self.user_dimension_reduction = nn.Sequential(*layers).to(self.device)

        self._init_weights()

    def _init_weights(self):
        """Initialize linear layer weights."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, uids, pids, creator_uids):
        """Compute pairwise influence probabilities from IDs."""
        u_target = self.user_dimension_reduction(self.user_embedding[uids]) 
        u_creator = self.user_dimension_reduction(self.user_embedding[creator_uids]) 
        c = self.post_dimension_reduction(self.post_embedding[pids]) 
        ut = u_target.unsqueeze(2)  
        uc = u_creator.unsqueeze(1)  
        diag_c = torch.diag_embed(c) 
        intermediate = torch.bmm(uc, diag_c)  
        ps = torch.sigmoid(torch.bmm(intermediate, ut)).squeeze()  
        return ps

    def predict(self, uids, content_embs, creator_uids): # directly use post embeddings
        """Compute probabilities using precomputed content embeddings."""
        u_target = self.user_dimension_reduction(self.user_embedding[uids]) 
        u_creator = self.user_dimension_reduction(self.user_embedding[creator_uids]) 
        c = self.post_dimension_reduction(content_embs) 
        ut = u_target.unsqueeze(2)  
        uc = u_creator.unsqueeze(1)  
        diag_c = torch.diag_embed(c)  
        intermediate = torch.bmm(uc, diag_c) 
        ps = torch.sigmoid(torch.bmm(intermediate, ut)).squeeze() 
        return ps


def train_model(model, optimizer, device, train_dataset, val_dataset, 
          epoch_num, batch_size, patience,
          save_file_name):
    """Train the predictor model with early stopping."""
    loss_func = nn.MSELoss(reduction='sum')
    best_epoch = 0
    best_loss = float('inf')
    epoch_not_improved = 0

    start_time = time.time()
    
    print('Start training')
    for epoch in range(epoch_num):
        dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, \
                                collate_fn=train_dataset.collate_batch)  
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, \
                                collate_fn=val_dataset.collate_batch)
        
        model.train()
        loss = 0
        train_instance_num = 0
        for i, batch in enumerate(dataloader):
            batch = batch_to_gpu(batch, device)
            p = model(batch['users'], batch['content'], batch['creator'])
            batch_loss = loss_func(p, batch['label'])
            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()
            loss += batch_loss.item()
            train_instance_num += len(batch['label'])
        print('Epoch', epoch, 'loss:', round(loss/train_instance_num, 4), 'time:', int(time.time()-start_time))
        
        if epoch % 1 == 0:
            model.eval()
            val_loss = 0
            val_instance = 0
            for _, val_batch in enumerate(val_dataloader):
                val_batch = batch_to_gpu(val_batch, device)
                val_p = model(val_batch['users'], val_batch['content'], val_batch['creator'])
                val_p = val_p.squeeze()
                val_loss += loss_func(val_p, val_batch['label']).item()
                val_instance += len(val_batch['label'])
            val_loss /= val_instance
            if val_loss <= best_loss:
                best_epoch = epoch
                best_loss = val_loss
                torch.save(model.state_dict(), save_file_name)
                print('Save best model at epoch', epoch, best_loss)
                epoch_not_improved = 0
            else:
                epoch_not_improved += 1
                if epoch_not_improved >= patience:
                    print("Early stopping at epoch", epoch)
                    break
    print("Best validation model saved at epoch:", best_epoch)
    return best_loss, best_epoch


def main(lr: float = 1e-4, 
        epoch_num: int = 1000, 
        batch_size: int = 1024,
        dataset_name: str ='weibo',
        patience: int = 20,
        seed: int = 42):
    """Train and evaluate the pairwise influence predictor.

    Args:
        lr: Learning rate.
        epoch_num: Maximum number of training epochs.
        batch_size: Training batch size.
        dataset_name: Dataset name used to load preprocessed data.
        patience: Early stopping patience.
        seed: Random seed.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    seed_everything(seed)

    if not os.path.exists('./predictor_weights'):
        os.makedirs('./predictor_weights')

    # ----------------- load dataset -----------------
    post_content_dict, post_author_dict, influence_dict = pickle_load('../datasets/{}/preprocessed/post_info.pkl'.format(dataset_name))
    post_dict, _, reverse_repost_dict = pickle_load('../datasets/{}/preprocessed/interaction_info.pkl'.format(dataset_name))
    follow_edges, two_hop_adj = pickle_load('../datasets/{}/preprocessed/network_info.pkl'.format(dataset_name))
    train, val, test = pickle_load('../datasets/{}/preprocessed/split_info.pkl'.format(dataset_name))
    
    one_hop_adj = defaultdict(list) # used for negative instances
    for u, v in follow_edges: # u follows v, v will influence u
        one_hop_adj[v].append(u)
    one_hop_adj = {k: list(set(v)) for k, v in one_hop_adj.items()}
    print("Finish loading dataset")

    two_hop_adj = one_hop_adj
    train_instances = prepare_instances(train, post_author_dict, reverse_repost_dict, two_hop_adj)
    val_instances = prepare_instances(val, post_author_dict, reverse_repost_dict, two_hop_adj)
    test_instances = prepare_instances(test, post_author_dict, reverse_repost_dict, two_hop_adj)
    print("Finish preparing dataset")

    train_instances = torch.tensor(train_instances)
    val_instances = torch.tensor(val_instances)
    test_instances = torch.tensor(test_instances)
    train_dataset = Dataset(users=train_instances[:, 0], creator=train_instances[:, 1], content=train_instances[:, 2],
                            label=train_instances[:, 3].float())
    val_dataset = Dataset(users=val_instances[:, 0], creator=val_instances[:, 1], content=val_instances[:, 2],
                            label=val_instances[:, 3].float())
    test_dataset = Dataset(users=test_instances[:, 0], creator=test_instances[:, 1], content=test_instances[:, 2],
                            label=test_instances[:, 3].float())
    print("Train, validation, test sizes:", len(train_instances), len(val_dataset), len(test_dataset))

    # ----------------- train -----------------
    prob_model = Prob_Model(dataset_name).to(device)
    optimizer = torch.optim.Adam(prob_model.parameters(), lr=lr, weight_decay=1e-4)
    print("Start training")
    save_file_name = './predictor_weights/{}.pt'.format(dataset_name)
    best_val, best_epoch = train_model(prob_model, optimizer, device, 
          train_dataset, val_dataset, epoch_num, 
          batch_size, patience, save_file_name)
    print("Finish training")

    # ----------------- test -----------------
    prob_model.load_state_dict(torch.load(save_file_name, weights_only=True))
    prob_model.eval()
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, \
                                collate_fn=test_dataset.collate_batch)
    test_loss = 0
    loss_func = nn.MSELoss(reduction='sum')
    instance_num = 0
    acc = 0
    pos_num = 0
    for _, test_batch in enumerate(test_dataloader):
        test_batch = batch_to_gpu(test_batch, device)
        test_p = prob_model(test_batch['users'], test_batch['content'], test_batch['creator'])
        test_loss += loss_func(test_p, test_batch['label']).item()
        instance_num += len(test_batch['label'])
        acc += ((test_p > 0.5).float() == test_batch['label']).sum().item()
        pos_num += ((test_p > 0.5).float() == 1).sum().item()
    print('Evaluation (test):', test_loss/instance_num, acc/instance_num)
    print('Positive num:', pos_num, 'Total num:', instance_num)

if __name__ == "__main__":
    from jsonargparse import CLI
    CLI(main)
