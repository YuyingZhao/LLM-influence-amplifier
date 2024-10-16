# Environment
Our code is mainly based on LitGPT for the LLM part. Please install this package using pip install 'litgpt[all]'.
We also use the sentence transformer for embedding extraction. Please install using pip install -U sentence-transformers.

# Description about dataset preprocessing
(1) Download the three files of weibo dataset from http://aminer.org/Influencelocality including:
root_content.txt: it contains the raw textual content of the posts;
weibo_network.txt: it contains the following relationships between users;
total.txt: it records the post and repost beheviors (e.g., who created which post and who reposted which post).
(2) Save them into datasets/weibo/src/, and run preprocessor.py.

# Description about pairwise influence predictor
Enter predictor directory:
run train_predictor.py to train the pairwise influence predictor and the weight will be automatically saved into ./predictor_weights/weibo.pt. To avoid computing the post embeddings repeatedly, the embeddings of posts will be computed once and saved in ./post_embeddings/weibo.pt.

# Desciption about prompt generation
Enter prompts directory:
run generate_prompt.py --use_structure='false' --flag=(zero_shot/few_shot_global/few_shot_personalized) for the content-centric prompts (1/2.1/2.2);
run generate_prompt.py --use_structure='true' --flag=(random_all/random_neighbor/influential_neighbor) for the structure-aware prompts focusing on sampling posts (3.0/3.1/3.2);
run generate_prompt.py --use_structure='true' --flag=(uniform/score) for the structure-aware prompts focusing on extracted interests (4.1/4.2). Before running this python scripts, prepare the interests using python neighbor_posts_extraction.py to obtain the neighborhood posts and interest_summarization.py to summarize the interests based on the neighborhood posts.
The generated prompts will be automatically saved in ./prompts.

# Description about LLM generation and evaluation
They are implemented in the same python script where the repsonse is generated based on the prompts and then evaluated through the pairwise prediction and MC simulation. Run python evaluation_influence.py to get the result.







