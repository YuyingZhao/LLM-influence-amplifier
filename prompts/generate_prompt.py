from utils import *

def main(dataset_name: str = 'weibo',
         seed: int = 42,
         use_structure: str = "true",
         flag: str = 'random_neighbor',
         num_hop: int = 1,
         num_sample: int = 10):
    if use_structure == 'true':
        print('structure-aware', flag)
    else:
        print('content-centric', flag)
    seed_everything(seed)

    # create ./prompts directory if it does not exist
    if not os.path.exists('./prompts'):
        os.makedirs('./prompts')

    # load data
    post_content_dict, post_author_dict, influence_dict = pickle_load('../datasets/{}/preprocessed/post_info.pkl'.format(dataset_name))
    post_dict, _, reverse_repost_dict = pickle_load('../datasets/{}/preprocessed/interaction_info.pkl'.format(dataset_name))
    follow_edges, two_hop_adj = pickle_load('../datasets/{}/preprocessed/network_info.pkl'.format(dataset_name))
    train, val, test = pickle_load('../datasets/{}/preprocessed/split_info.pkl'.format(dataset_name))
    repost_dict = defaultdict(list)
    for pid, uids in reverse_repost_dict.items():
        for uid in uids:
            repost_dict[uid].append(pid)

    degrees = defaultdict(int)
    for u, v in follow_edges: # u follows v, v influences u
        degrees[v] += 1
    if num_hop == 1:
        adj = defaultdict(list)
        for u, v in follow_edges: # u follows v, v influences u
            adj[v].append(u)
        adj = {k: v for k, v in adj.items() if len(v) > 0}
    elif num_hop == 2:
        adj = two_hop_adj

    test = [t for t in test if 'error' not in post_content_dict[t]] # remove error posts
    pids = test[:200]
    pid_to_author = post_author_dict
    all_post_list = list(post_content_dict.keys())

    if flag == 'few_shot_global':
        # prepare popular and unpopular posts in the training dataset
        influence = np.array([influence_dict[pid] for pid in train])
        threshold_num = int(len(train)*0.2)
        popular_posts = [train[t] for t in np.argsort(influence)[::-1][:threshold_num]]
        unpopular_posts = [train[t] for t in np.argsort(influence)[:threshold_num]]
        print('Average influence of popular posts:', np.mean([influence_dict[t] for t in popular_posts]).round(2))
        print('Average influence of unpopular posts:', np.mean([influence_dict[t] for t in unpopular_posts]).round(2))
        popular_content = random.choices(list(popular_posts), k=2)
        unpopular_content = random.choices(list(unpopular_posts), k=2)
        popular_content = ["'"+post_content_dict[t]+"'" for t in popular_content]
        unpopular_content = ["'"+post_content_dict[t]+"'" for t in unpopular_content]
    elif flag == 'few_shot_personalized':
        # prepare popular and unpopular posts in the training dataset
        influence = np.array([influence_dict[pid] for pid in train])
        threshold_num = int(len(train)*0.2)
        popular_posts = [train[t] for t in np.argsort(influence)[::-1][:threshold_num]]
        unpopular_posts = [train[t] for t in np.argsort(influence)[:threshold_num]]
        print('Average influence of popular posts:', np.mean([influence_dict[t] for t in popular_posts]).round(2))
        print('Average influence of unpopular posts:', np.mean([influence_dict[t] for t in unpopular_posts]).round(2))

        # calculate the similarity between the user's posts and the popular/unpopular posts
        # load embedding from file
        post_embedding = torch.load('../predictor/post_embeddings/{}.pt'.format(dataset_name), weights_only=True)
        user_post_embeddings = post_embedding[pids]
        popular_post_embeddings = post_embedding[popular_posts]
        unpopular_post_embeddings = post_embedding[unpopular_posts]
        user_popular_sim = torch.matmul(user_post_embeddings, popular_post_embeddings.T) # shape: (num_user_posts, num_popular_posts)
        user_unpopular_sim = torch.matmul(user_post_embeddings, unpopular_post_embeddings.T)
        print("Shapes:", user_popular_sim.shape, user_unpopular_sim.shape) # (200, 733)
        top2_indices_popular = torch.topk(user_popular_sim, 2, dim=1).indices
        top2_indices_unpopular = torch.topk(user_unpopular_sim, 2, dim=1).indices
        popular_idx_to_id = {k: v for k, v in enumerate(popular_posts)}
        unpopular_idx_to_id = {k: v for k, v in enumerate(unpopular_posts)}

    if use_structure == 'true' and (flag == 'uniform' or flag == 'score'):
        with open('./interests/interests_{}_{}_{}_{}.pkl'.format(dataset_name, flag, str(num_hop), str(num_sample)), 'rb') as f:
            pid_interest_dict = pickle.load(f)
            
    prompt_dict = {}
    for i, pid in enumerate(pids):
        uid = pid_to_author[pid]
        original_content = post_content_dict[pid]

        if use_structure == 'true' and (flag == 'uniform' or flag == 'score'):
            interest = pid_interest_dict[pid]
            prompt_dict[pid] = "Imagine you have a piece of text that you want to share on social media, but you want to ensure it \
            catches the maximum attention and engagement from your audience. Your task is to creatively revise the original text to make it more engaging and more likely to be shared widely. \
            The revised version should retain the core message but be optimized to resonate with social media trends and audience preferences. \
            Your goal is to transform the text into a revised one that can lead to a larger cascade of shares, likes, and comments. Your audience has the following interest: {}. \
            Based on their preferences, now transform the text for higher popularity. ### Input text={}. ### Revised text=".format(interest, original_content)
        elif use_structure == 'true':
            if flag == 'random_neighbor':
                neighbors = adj[uid]
                neighbor_posts = []
                for neighbor in neighbors:
                    neighbor_reposts = repost_dict[neighbor]
                    # randomly select a post from the neighbor's reposts
                    if len(neighbor_reposts) == 0:
                        continue
                    neighbor_post = np.random.choice(neighbor_reposts)
                    neighbor_posts.append("'"+post_content_dict[neighbor_post]+"'")
                # randomly select 3 posts from the neighbor's reposts
                neighbor_posts = np.random.choice(neighbor_posts, min(3, len(neighbor_posts)), replace=False)
            elif flag == 'influential_neighbor':
                neighbors = adj[uid]
                neighbor_degree = [degrees[n] for n in neighbors]
                neighbor = neighbors[np.argmax(neighbor_degree)]
                neighbor_reposts = repost_dict[neighbor]
                selected_posts = np.random.choice(neighbor_reposts, min(3, len(neighbor_reposts)))
                neighbor_posts = ["'"+post_content_dict[t]+"'" for t in selected_posts]
            elif flag == 'random_all':
                random_post_ids = np.random.choice(all_post_list, 3)
                neighbor_posts = []
                for i in random_post_ids:
                    neighbor_posts.append("'"+post_content_dict[i]+"'")
            neighbor_posts = ", ".join(neighbor_posts)
            neighbor_posts += '.'

            prompt_dict[pid] = "Imagine you have a piece of text that you want to share on social media, but you want to ensure it \
            catches the maximum attention and engagement from your audience. Your task is to creatively revise the original text to make it more engaging and more likely to be shared widely. \
            The revised version should retain the core message but be optimized to resonate with social media trends and audience preferences. \
            Your goal is to transform the text into a revised one that can lead to a larger cascade of shares, likes, and comments. Your audience has interacted with the following posts {}. \
            Based on their preferences, now transform the text for higher popularity. ### Input text={}. ### Revised text=".format(neighbor_posts, original_content)
        else:
            if flag == 'zero_shot':
                prompt_dict[pid] = "Imagine you have a piece of text that you want to share on social media, but you want to ensure it \
                catches the maximum attention and engagement from your audience. Your task is to creatively revise the original text to make it more engaging and more likely to be shared widely. \
                The revised version should retain the core message but be optimized to resonate with social media trends and audience preferences. \
                Your goal is to transform the text into a revised one that can lead to a larger cascade of shares, likes, and comments. ### Input text={}. ### Revised text=".format(original_content)
            elif flag == 'few_shot_global': # the examples of popular and unpopular posts are from the entire training dataset and they are not relevant to the original content
                prompt_dict[pid] = "Imagine you have a piece of text that you want to share on social media, but you want to ensure it \
                catches the maximum attention and engagement from your audience. Your task is to creatively revise the original text to make it more engaging and more likely to be shared widely. \
                The revised version should retain the core message but be optimized to resonate with social media trends and audience preferences. \
                Your goal is to transform the text into a revised one that can lead to a larger cascade of shares, likes, and comments. \
                Following are the examples of popular posts: {}. Following are the examples of unpopular posts: {}. \
                Based on the instructions and examples, now transform the text for higher popularity. ### Input text={}. ### Revised text=".format(popular_content, unpopular_content, original_content)
            elif flag == 'few_shot_personalized':
                popular_idx = top2_indices_popular[i].tolist()
                unpopular_idx = top2_indices_unpopular[i].tolist()
                popular_posts = [popular_idx_to_id[t] for t in popular_idx]
                unpopular_posts = [unpopular_idx_to_id[t] for t in unpopular_idx]
                popular_content = ["'"+post_content_dict[t]+"'" for t in popular_posts]
                unpopular_content = ["'"+post_content_dict[t]+"'" for t in unpopular_posts]
                prompt_dict[pid] = "Imagine you have a piece of text that you want to share on social media, but you want to ensure it \
                catches the maximum attention and engagement from your audience. Your task is to creatively revise the original text to make it more engaging and more likely to be shared widely. \
                The revised version should retain the core message but be optimized to resonate with social media trends and audience preferences. \
                Your goal is to transform the text into a revised one that can lead to a larger cascade of shares, likes, and comments. \
                Following are the examples of popular posts: {}. Following are the examples of unpopular posts: {}. \
                Based on the instructions and examples, now transform the text for higher popularity. ### Input text={}. ### Revised text=".format(popular_content, unpopular_content, original_content)
            
    # save prompt
    if use_structure == 'true' and (flag == 'uniform' or flag == 'score'):
        with open('./prompts/structure_interests_{}_{}_{}_{}.pkl'.format(dataset_name, flag, str(num_hop), str(num_sample)), 'wb') as f:
            pickle.dump(prompt_dict, f)
    elif use_structure == 'true':
        with open('./prompts/structure_posts_{}_{}_{}.pkl'.format(dataset_name, flag, str(num_hop)), 'wb') as f:
            pickle.dump(prompt_dict, f)
    else:
        with open('./prompts/no_structure_{}_{}.pkl'.format(dataset_name, flag), 'wb') as f:
            pickle.dump(prompt_dict, f)
    print('Prompt generation done!')


if __name__ == "__main__":
    from jsonargparse import CLI
    CLI(main)