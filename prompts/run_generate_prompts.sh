# content-centric
# for flag in 'zero_shot' 'few_shot_global' 'few_shot_personalized'
# do
#     python generate_prompt.py --dataset_name='weibo' --use_structure='false' --flag=$flag
# done

# structure-aware, sampled posts
# for flag in 'random_all' 'random_neighbor' 'influential_neighbor'
# do
#     python generate_prompt.py --dataset_name='weibo' --use_structure='true' --flag=$flag
# done

# structure-aware, extracted interests
# extract the posts from neighbors
# python neighbor_posts_extraction.py --dataset_name='weibo' --sample_flag='uniform' --num_hop=1
# python neighbor_posts_extraction.py --dataset_name='weibo' --sample_flag='score' --num_hop=1
# summarize the interests
# python interest_summarization.py --dataset_name='weibo' --sample_flag='uniform' --num_hop=1
# python interest_summarization.py --dataset_name='weibo' --sample_flag='score' --num_hop=1
for flag in 'uniform' 'score'
do
    python generate_prompt.py --dataset_name='weibo' --use_structure='true' --flag=$flag
done