"""Export: ordinary users' handles are masked before anything is published."""
from post_scorer.export import mask_private_handles


def test_masks_fan_handle_and_truth_social_user_links():
    assert mask_private_handles('"@fan123: You are great!" Thanks') == '"@[fan account]: You are great!" Thanks'
    assert mask_private_handles("https://truthsocial.com/users/someone/statuses/1") == "https://truthsocial.com/users/[user]/statuses/1"
    assert mask_private_handles("https://truthsocial.com/@someone/1") == "https://truthsocial.com/users/[user]/1"


def test_keeps_mentions_in_his_own_words():
    text = "Thank you @FoxNews and @JohnDoe for the interview!"
    assert mask_private_handles(text) == text
