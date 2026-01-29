"""Summarize audience interests from neighboring posts."""

from transformers import AutoModelForCausalLM, AutoTokenizer
from utils import *


def main(
    dataset_name: str = 'weibo',
    num_hop: int = 1,
    sample_post_num: int = 10,
    sample_flag: str = 'uniform'
):
    """Generate interest summaries based on neighboring posts.

    Args:
        dataset_name: Name of the dataset to use.
        num_hop: Hop count for neighborhood selection.
        sample_post_num: Number of neighboring posts to sample.
        sample_flag: Sampling strategy flag.
    """
    seed_everything(42)

    device = "cuda" # the device to load the model onto

    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2-7B-Instruct",
        torch_dtype="auto",
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-7B-Instruct")

    # load data
    post_content_dict, post_author_dict, influence_dict = pickle_load('../datasets/{}/preprocessed/post_info.pkl'.format(dataset_name))
    with open('./interests/neighbor_posts_{}_{}_{}_{}.pkl'.format(dataset_name, sample_flag, str(num_hop), str(sample_post_num)), 'rb') as f:
        pid_posts_dict = pickle.load(f)

    pid_interests = dict()
    for pid, posts in pid_posts_dict.items():
        t = [post_content_dict[pid] for pid in posts]
        t = ["'"+x+"'" for x in t]
        t = ", ".join(t)
        
        prompt = "Your audience have interacted with the following posts:{}. Summarize the interest of your audience based on their interactions and output the interests seperated by ';', without any explanation or introduction.".format(t)
        messages = [
            {"role": "system", "content": "You are a content creator in social media, and you want to ensure your content catches the maximum attention and engagement from your audience."},
            {"role": "user", "content": prompt}
        ]

        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        model_inputs = tokenizer([text], return_tensors="pt").to(device)

        generated_ids = model.generate(
            model_inputs.input_ids,
            attention_mask=model_inputs.attention_mask,
            max_new_tokens=512,
            temperature=0.001
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        pid_interests[pid] = response
        print(response)
    # save the interests
    with open('./interests/interests_{}_{}_{}_{}.pkl'.format(dataset_name, sample_flag, str(num_hop), str(sample_post_num)), 'wb') as f:
        pickle.dump(pid_interests, f)

if __name__ == "__main__":
    from jsonargparse import CLI
    CLI(main)
