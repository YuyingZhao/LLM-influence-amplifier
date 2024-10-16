from utils import *
from torch_geometric.utils import degree
import torch.nn.functional as F
    
def main(
    dataset_name: str = 'weibo',
    num_hop: int = 1,
    sample_post_num: int = 10,
    sample_flag: str = 'uniform'
):
    seed_everything(42)
    # load data
    post_content_dict, post_author_dict, influence_dict = pickle_load('../datasets/{}/preprocessed/post_info.pkl'.format(dataset_name))
    post_dict, _, reverse_repost_dict = pickle_load('../datasets/{}/preprocessed/interaction_info.pkl'.format(dataset_name))
    follow_edges, two_hop_adj = pickle_load('../datasets/{}/preprocessed/network_info.pkl'.format(dataset_name))
    train, val, test = pickle_load('../datasets/{}/preprocessed/split_info.pkl'.format(dataset_name))

    one_hop_adj = defaultdict(list)
    for u, v in follow_edges: # u follows v, v influences u
        one_hop_adj[v].append(u)

    repost_dict = defaultdict(list)
    for pid, uids in reverse_repost_dict.items():
        for uid in uids:
            repost_dict[uid].append(pid)
    # update to only include training posts
    for uid in repost_dict:
        repost_dict[uid] = [pid for pid in repost_dict[uid] if pid in train]

    test_dict = pickle_load('./prompts/no_structure_{}_{}.pkl'.format(dataset_name, 'zero_shot')) # just used to obtain the test pids
    test_pids = list(test_dict.keys())
    print('number of test posts:', len(test_pids))

    if sample_flag == 'score':
        # calculate the importance through message passing and then use softmax to sample
        users = []
        for u, v in follow_edges:
            users.append(u)
            users.append(v)
        user_num = max(users) + 1

        # one_hop_adj[u]: users that will be impacted by u
        follow_edges = follow_edges + [(i, i) for i in range(user_num)] # add self loop to follow_edges
        edge_index = torch.tensor(follow_edges).t() # 2, edge_num
        row, col = edge_index
        deg = degree(col)
        deg_sqrt = deg.pow(0.5)

        all_users = list(range(user_num))
        h_emb = torch.eye(user_num)
        for hop in range(num_hop):
            h_emb_new = torch.zeros_like(h_emb)
            for u in all_users:
                for i in (one_hop_adj[u]+[u]):
                    h_emb_new[u] += (deg_sqrt[u].item())*(deg_sqrt[i].item())*(h_emb[i])
            h_emb = h_emb_new
        # set diagonal to 0 to avoid sampling the creator
        for i in range(user_num):
            h_emb[i][i] = 0
        h_emb = h_emb/5.0 # temperature
        sample_prob = F.softmax(h_emb, dim=1)

    pid_posts_dict = defaultdict(list) # store the sampled posts from the neighborhood
    for i, pid in enumerate(test_pids):
        author = post_author_dict[pid]

        if num_hop == 1:
            neighbors = one_hop_adj[author]
        elif num_hop == 2:
            neighbors = two_hop_adj[author]
        
        neighbor_interested_posts = set()
        threshold = sample_post_num * 10
        if sample_flag == 'uniform':
            while len(neighbor_interested_posts) < sample_post_num:
                threshold -= 1
                neighbor = random.choice(neighbors)
                neighbor_reposts = repost_dict[neighbor]
                if threshold == 0: # to avoid infinite loop
                    break
                if len(neighbor_reposts) == 0:
                    continue
                repost = random.choice(neighbor_reposts)
                if repost in neighbor_interested_posts:
                    continue
                neighbor_interested_posts.add(repost)
        elif sample_flag == 'score':
            while len(neighbor_interested_posts) < sample_post_num:
                threshold -= 1
                neighbor = random.choices(neighbors, sample_prob[author][neighbors].tolist())[0]
                neighbor_reposts = repost_dict[neighbor]
                if threshold == 0:
                    break
                if len(neighbor_reposts) == 0:
                    continue
                repost = random.choice(neighbor_reposts)
                if repost in neighbor_interested_posts:
                    continue
                neighbor_interested_posts.add(repost)
        pid_posts_dict[pid] = neighbor_interested_posts
    
    # to avoid repeated storing the content, will only save the ids

    if not os.path.exists('./interests'):
        os.makedirs('./interests')
    with open('./interests/neighbor_posts_{}_{}_{}_{}.pkl'.format(dataset_name, sample_flag, str(num_hop), str(sample_post_num)), 'wb') as f:
        pickle.dump(pid_posts_dict, f)
    print('finish saving the posts from the neighbors', sample_flag, num_hop, sample_post_num)


if __name__ == "__main__":
    from jsonargparse import CLI
    CLI(main)