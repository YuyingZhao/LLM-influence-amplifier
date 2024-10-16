import numpy as np
from collections import defaultdict
import random
import pickle
from datetime import datetime
import os
import torch

def seed_everything(seed):
    np.random.seed(seed)
    random.seed(seed)

def load_weibo():
    # transform raw data into structured format
    content_filename = './weibo/src/root_content.txt'
    post_repost_filename = './weibo/src/total.txt'

    repost_time_dict = defaultdict(list) # record the repost time, values are a list of tuple pid and time
    post_time_dict = defaultdict(list) # record the post time, key is pid, value is the time

    # load post content
    post_content_dict = {} 
    with open(content_filename, 'r', encoding='gbk') as file:
        lines = file.readlines()
        for i in range(0, len(lines), 2):
            pid = lines[i].strip()
            post_content = lines[i + 1].strip()
            post_content_dict[pid] = post_content
    print('Finish loading post content:', len(post_content_dict))
    
    # load post author/post/repost
    post_author_dict = dict()
    repost_dict = defaultdict(list) 
    reverse_repost_dict = defaultdict(list)
    influence_dict = defaultdict(int)
    with open(post_repost_filename, 'r') as file:
        lines = file.readlines()
        for i in range(0, len(lines), 2):
            original_line = lines[i].strip().split()
            original_post_id = original_line[0]
            original_user_id = original_line[2]
            retweet_num = int(original_line[3])
            if retweet_num < 1 or original_post_id not in post_content_dict.keys(): # filter out those with no retweet or no content
                continue
            influence_dict[original_post_id] = retweet_num
            post_author_dict[original_post_id] = original_user_id
            repost_line = lines[i + 1].strip().split()
            for j in range(0, len(repost_line), 2):
                repost_user_id = repost_line[j] 
                # the user who repost the original post
                repost_dict[repost_user_id].append(original_post_id)
                reverse_repost_dict[original_post_id].append(repost_user_id)
    post_dict = defaultdict(list) 
    # key is uid and values are pids that the user posts
    for pid, uid in post_author_dict.items():
        post_dict[uid].append(pid)
    print('Finish loading interactions:', len(post_author_dict), len(influence_dict), len(post_dict), len(repost_dict))

    posts = set(list(post_author_dict.keys()))
    post_content_dict = {k: v for k, v in post_content_dict.items() if k in posts}
    print('Number of posts:', len(posts))
    print('Number of users:', len(post_dict))

    post_time_dict = defaultdict(list) # key is uid, values are the timestamps for posts
    repost_time_dict = defaultdict(list) # key is uid, values are the timestamps for reposts
    with open(post_repost_filename, 'r') as file:
        lines = file.readlines()
        for i in range(0, len(lines), 2):
            original_line = lines[i].strip().split()
            original_time = original_line[1]
            original_pid = original_line[0]

            if original_pid not in posts:
                continue
            # the original time format need to be transformed into integer type using Unix timestamp for further sorting
            date_obj = datetime.strptime(original_time, '%Y-%m-%d-%H:%M:%S')
            original_time = int(date_obj.timestamp())
            post_time_dict[original_pid] = original_time

            repost_line = lines[i + 1].strip().split()
            for j in range(0, len(repost_line), 2):
                repost_user_id = repost_line[j] 
                repost_time = repost_line[j+1]
                repost_time_dict[repost_user_id].append((original_pid, repost_time))

    return post_content_dict, post_author_dict, post_dict, repost_dict, reverse_repost_dict, influence_dict, post_time_dict, repost_time_dict

def load_weibo_network(all_users):
    follow_filename = './weibo/src/weibo_network.txt'
    follow_edges = []
    with open(follow_filename, 'r') as file:
        N, M = map(int, file.readline().split())
        for _ in range(N):
            line = file.readline().split()
            follower, k = line[0], int(line[1])
            if follower in all_users:
                for i in range(2, 2 + 2 * k, 2):
                    followee = line[i] # the users that follower follows
                    if followee in all_users:
                        follow_edges.append([follower, followee]) # (u, v) means u follows v
    follow_edges = np.array(follow_edges)
    print("Edge number in follow network:", follow_edges.shape)
    print('Number of src (followed by others):', len(set(follow_edges[:, 0])))
    print('Number of target (following others):', len(set(follow_edges[:, 1])))

    user_index = {u: i for i, u in enumerate(all_users)}
    influence_adj = torch.zeros(len(all_users), len(all_users)).to(device='cuda')
    for u, v in follow_edges: # u follows v
        influence_adj[user_index[v], user_index[u]] = 1
    two_hop_adj = influence_adj @ influence_adj
    two_hop_adj = torch.logical_or(influence_adj, two_hop_adj).int()
    two_hop_adj = two_hop_adj.cpu().numpy()
    two_hop_adj = {i: list(np.where(two_hop_adj[i])[0]) for i in range(len(two_hop_adj))}

    follow_edges = [[user_index[u], user_index[v]] for u, v in follow_edges]
    return follow_edges, two_hop_adj, user_index

def dump_data(data, filename):
    with open(filename, 'wb') as f:
        pickle.dump(data, f)

class preprocessor:
    def __init__(self, dataset_name, dataset_save_prefix):
        self.dataset_name = dataset_name
        self.dataset_save_prefix = dataset_save_prefix
        if dataset_name == 'weibo':
            self.post_content_dict, self.post_author_dict, self.post_dict, self.repost_dict, self.reverse_repost_dict, self.realworld_influence_dict, \
                self.post_time_dict, self.repost_time_dict = load_weibo()
            self.influence_dict = {k: len(v) for k, v in self.reverse_repost_dict.items()}
            user_subset = self.extract_dense_subset(topk=20)
            self.follow_edges, self.two_hop_adj, self.user_index = load_weibo_network(user_subset)
            post_subset = self.prune_cold_start(user_subset)

            # index and save the processed data
            all_posts = []
            for u, posts in self.post_dict.items():
                all_posts.extend(posts)
            all_posts = set(all_posts)
            post_subset = set(post_subset).intersection(all_posts)
            self.post_index = {k: i for i, k in enumerate(all_posts)}
            self.post_content_dict = {self.post_index[k]: v for k, v in self.post_content_dict.items() if k in all_posts}
            self.post_author_dict = {self.post_index[k]: self.user_index[v] for k, v in self.post_author_dict.items() if k in all_posts}
            self.post_dict = {self.user_index[k]: [self.post_index[pid] for pid in v if pid in self.post_index] for k, v in self.post_dict.items() if k in user_subset}
            self.repost_dict = {self.user_index[k]: [self.post_index[pid] for pid in v if pid in self.post_index] for k, v in self.repost_dict.items() if k in user_subset}
            self.reverse_repost_dict = {self.post_index[k]: v for k, v in self.reverse_repost_dict.items() if k in post_subset}
            self.realworld_influence_dict = {self.post_index[k]: v for k, v in self.realworld_influence_dict.items() if k in all_posts}

            dump_data((self.post_content_dict, self.post_author_dict, self.realworld_influence_dict), self.dataset_save_prefix+'/post_info.pkl')
            dump_data((self.post_dict, self.repost_dict, self.reverse_repost_dict), self.dataset_save_prefix+'/interaction_info.pkl')
            dump_data((self.follow_edges, self.two_hop_adj), self.dataset_save_prefix+'/network_info.pkl')

            # split data into train/val/test
            posts = list(post_subset)
            posts = [self.post_index[k] for k in posts]
            random.shuffle(posts)
            train_posts, val_posts, test_posts = posts[:int(0.6*len(posts))], posts[int(0.6*len(posts)):int(0.8*len(posts))], posts[int(0.8*len(posts)):]
            dump_data((train_posts, val_posts, test_posts), self.dataset_save_prefix+'/split_info.pkl')
            print('Number of users:', len(self.user_index))
            print('Number of posts:', len(self.post_index), len(post_subset))
            print('Follow network:', len(self.follow_edges))
            print('Average post number per user:', np.mean([len(v) for k, v in self.post_dict.items()]), np.min([len(v) for k, v in self.post_dict.items()]), np.max([len(v) for k, v in self.post_dict.items()]))
            print('Average repost number per user:', np.mean([len(v) for k, v in self.repost_dict.items()]), np.min([len(v) for k, v in self.repost_dict.items()]), np.max([len(v) for k, v in self.repost_dict.items()]))
            print('Average repost number per item:', np.mean([len(v) for k, v in self.reverse_repost_dict.items()]), np.min([len(v) for k, v in self.reverse_repost_dict.items()]), np.max([len(v) for k, v in self.reverse_repost_dict.items()]))
            print('Split:', len(train_posts), len(val_posts), len(test_posts))
            print('Finish saving preprocessed data')

    def extract_dense_subset(self, topk):
        # obtain the users who (1) are involved in the topk posts within multiple hops (2) have at least one post and repost behavior
        topk_posts = sorted(self.influence_dict.items(), key=lambda x: x[1], reverse=True)[:topk]
        users = [self.post_author_dict[k] for k, v in topk_posts] # initial seed
        users = list(set(users))
        print('Initial user set:', len(users))
        total_users = []
        total_users.extend(users)
        visited_users = []
        # relatively active users with post and repost behaviors
        post_user_set = set([k for k, v in self.post_dict.items() if len(v)>=2])
        repost_user_set = set([k for k, v in self.repost_dict.items() if len(v)>=2])
        post_repost_user_set = post_user_set.intersection(repost_user_set)
        for i in range(2):
            new_users = []
            for u in users:
                # the authors that u has reposted
                if u in self.repost_dict.keys():
                    repost_list = self.repost_dict[u]
                    new_users.extend([self.post_author_dict[pid] for pid in repost_list])
                # the users who repost u's post
                if u in self.post_dict.keys():
                    pids = self.post_dict[u]
                    for pid in pids:
                        if pid in self.reverse_repost_dict.keys():
                            new_users.extend(self.reverse_repost_dict[pid])
            new_users = list(set(new_users).intersection(post_repost_user_set)) # at least one post and repost for each user
            total_users.extend(new_users)
            visited_users.extend(users)
            users = list(set(new_users)-set(visited_users))
            print("Within", i, "hops - Users involved in topk posts:", len(set(total_users)))
        return set(total_users).intersection(post_repost_user_set)

    def prune_cold_start(self, user_subset):
        # obtain posts with at least 5 repost behaviors within the user subset
        pruned_reverse_repost_dict = {}
        for post, repost_users in self.reverse_repost_dict.items():
            author = self.post_author_dict[post]
            if author not in user_subset:
                continue
            repost_users = set([self.user_index[t] for t in repost_users if t in self.user_index]).intersection(set(self.two_hop_adj[self.user_index[author]]))
            if len(repost_users) >= 5:
                pruned_reverse_repost_dict[post] = list(repost_users)
        
        # update post to maintain a shorter length
        self.post_dict = {k: v for k, v in self.post_dict.items() if k in user_subset}
        post_with_repost = set(pruned_reverse_repost_dict.keys())
        for user, posts in self.post_dict.items():
            t = list(set(posts).intersection(post_with_repost))
            if len(t) != 0:
                self.post_dict[user] = random.sample(t, min(len(t), 5))
            else:
                self.post_dict[user] = random.sample(posts, min(len(posts), 5))

        self.reverse_repost_dict = pruned_reverse_repost_dict
        # posts with at least one repost behavior within the following network
        post_subset = set(self.reverse_repost_dict.keys())
        # these posts have the following two properties: 1) the author is in the user subset; 2) the post has at least one repost behavior within the 2 hop following network of the author
        print('Number of users:', len(user_subset))
        print('Number of posts:', len(post_subset))
        return post_subset

def main(dataset_name: str ='weibo',
        seed: int = 42):
    seed_everything(seed)
    dataset_save_prefix = './'+dataset_name+'/preprocessed/'
    if not os.path.exists(dataset_save_prefix):
        os.makedirs(dataset_save_prefix)
    preprocessor(dataset_name, dataset_save_prefix)


if __name__ == "__main__":
    from jsonargparse.cli import CLI
    CLI(main)

