import numpy as np
from collections import defaultdict
import torch
from sentence_transformers import SentenceTransformer
from litgpt import LLM
import sys
sys.path.insert(1, 'the path to the folder containing train_predictor.py [predictor]')
from train_predictor import *

class influence_estimator():
    def __init__(self, adj, dataset_name, mc_num):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.adj = adj
        self.mc_num = mc_num
        self.dataset_name = dataset_name

        # load predictor model
        prob_model = Prob_Model(dataset_name=dataset_name)
        prob_model.to(self.device)
        file_name = '../predictor/predictor_weights/{}.pt'.format(dataset_name)
        prob_model.load_state_dict(torch.load(file_name, weights_only=True))
        prob_model.eval()
        self.model = prob_model

        self.max_hop = 2

        self.sentence_model = SentenceTransformer('sentence-transformers/distiluse-base-multilingual-cased-v2')
        self.sentence_model.eval()
        self.sentence_model.to(self.device)
        for parameters in self.sentence_model.parameters():
            parameters.requires_grad = False

        # obtain the degree of each node
        self.degree = defaultdict(int)
        for u, v in adj.items(): # u will impact v
            for t in v:
                self.degree[t] += 1 # record how many users will impact t
        self.degree = {k: v for k, v in self.degree.items()}
        
    def estimate(self, node_id, content_id): 
        spread = []
        for i in range(self.mc_num):
            new_active, A = [(0, node_id)], [node_id]
            while new_active:
                new_ones = []
                for (h, node) in new_active:
                    if h < self.max_hop:
                        if h!= 0:
                            continue
                        if node not in self.adj:
                            continue
                        neighbors = self.adj[node]
                        if len(neighbors) == 0:
                            continue
                        probs = self.model(neighbors, [content_id]*len(neighbors), [node]*len(neighbors))
                        probs = probs.squeeze().tolist()
                        success = np.random.uniform(0,1,len(neighbors)) < probs
                        new_ones += [(h+1, t) for t in list(np.extract(success, neighbors))]
                A += list(set([t[1] for t in new_ones]) - set(A))
                new_active = new_ones                    
            spread.append(A)
        return spread

    def estimate_post(self, node_id, content): 
        content_emb = self.sentence_model.encode(content)
        content_emb = torch.from_numpy(content_emb).to(self.device)
        spread = []
        for i in range(self.mc_num):
            new_active, A = [(0, node_id)], [node_id]
            while new_active:
                new_ones = []
                for (h, node) in new_active:
                    if h < self.max_hop:
                        if h!= 0:
                            continue
                        if node not in self.adj:
                            continue
                        neighbors = self.adj[node]
                        if len(neighbors) == 0:
                            continue
                        content_embeddings = torch.stack([content_emb]*len(neighbors), dim=0)
                        probs = self.model.predict(neighbors, content_embeddings, [node]*len(neighbors))
                        probs = probs.squeeze().tolist()
                        success = np.random.uniform(0,1,len(neighbors)) < probs
                        new_ones += [(h+1, t) for t in list(np.extract(success, neighbors))]
                A += list(set([t[1] for t in new_ones]) - set(A))
                new_active = new_ones                    
            spread.append(A)
        return spread

def main(dataset_name: str = 'weibo',
         mc_num: int = 20,
         seed: int = 42,
         llm_name: str = 'phi-2',
         flag: str = 'random_neighbor',
         use_structure: str = 'true',
         num_hop: int = 1,
         num_sample: int = 10):
    seed_everything(seed)
    if not os.path.exists('results'):
        os.makedirs('results')
    if use_structure == 'true' and (flag == 'uniform' or flag == 'score'):
        print(llm_name, 'structure-aware-extracted-interests', flag, num_hop, num_sample)
        save_file_prefix = './results/structure_interests_{}_{}_{}_{}'.format(llm_name, flag, num_hop, num_sample)
    elif use_structure == 'true':
        print(llm_name, 'structure-aware-sampled-posts', flag, num_hop)
        save_file_prefix = './results/structure_posts_{}_{}_{}'.format(llm_name, flag, num_hop)
    else:
        print(llm_name, 'content-centric', flag)
        save_file_prefix = './results/no_structure_{}_{}'.format(llm_name, flag)

    # litgpt download list
    if llm_name == 'phi-2':
        llm = LLM.load("../../checkpoints/microsoft/phi-2")
    elif llm_name == 'phi-3':
        llm = LLM.load("../../checkpoints/microsoft/Phi-3-mini-4k-instruct")
    elif llm_name == 'mistral':
        llm = LLM.load("../../checkpoints/mistralai/Mistral-7B-Instruct-v0.3")
    elif llm_name == 'llama2_chat':
        llm = LLM.load("../../checkpoints/meta-llama/Llama-2-7b-chat-hf")

    # load data
    post_content_dict, post_author_dict, influence_dict = pickle_load('../datasets/{}/preprocessed/post_info.pkl'.format(dataset_name))
    post_dict, _, reverse_repost_dict = pickle_load('../datasets/{}/preprocessed/interaction_info.pkl'.format(dataset_name))
    follow_edges, two_hop_adj = pickle_load('../datasets/{}/preprocessed/network_info.pkl'.format(dataset_name))
    train, val, test = pickle_load('../datasets/{}/preprocessed/split_info.pkl'.format(dataset_name))    
    repost_dict = defaultdict(list)
    for pid, uids in reverse_repost_dict.items():
        for uid in uids:
            repost_dict[uid].append(pid)

    adj = defaultdict(list)
    for u, v in follow_edges: # u follows v, v influences u
        adj[v].append(u)
    adj = {k: v for k, v in adj.items() if len(v) > 0}

    if use_structure == 'true' and (flag == 'uniform' or flag == 'score'):
        test_dict = pickle_load('../prompts/prompts/structure_interests_{}_{}_{}_{}.pkl'.format(dataset_name, flag, str(num_hop), str(num_sample)))
    elif use_structure == 'true':
        test_dict = pickle_load('../prompts/prompts/structure_posts_{}_{}_{}.pkl'.format(dataset_name, flag, str(num_hop)))
    else:
        test_dict = pickle_load('../prompts/prompts/no_structure_{}_{}.pkl'.format(dataset_name, flag))

    time_start = time.time()
    estimator = influence_estimator(adj, dataset_name=dataset_name, mc_num=mc_num)
    influence_spreads_revised = []
    revised_content_dict = dict()
    for pid, prompt in test_dict.items():
        uid = post_author_dict[pid]
        revised_content = llm.generate(prompt, temperature=0.001, max_new_tokens=200)
        revised_content_dict[pid] = revised_content
        predicted = estimator.estimate_post(uid, revised_content)
        influence_spread_revised = np.mean([len(t) for t in predicted])
        influence_spreads_revised.append(influence_spread_revised)
    print("time:", time.time()-time_start)

    # save influence spread results
    with open(save_file_prefix+'_influence_spread.pkl', 'wb') as f:
        pickle.dump(influence_spreads_revised, f)
    with open(save_file_prefix+'_content.pkl', 'wb') as f:
        pickle.dump(revised_content_dict, f)

    print("average spread revised:", round(np.mean(influence_spreads_revised), 2))
    print("spread improvement:", round((np.mean(influence_spreads_revised) -  121.92)/121.92*100, 2), "%")
    print()


if __name__ == "__main__":
    from jsonargparse import CLI
    CLI(main)