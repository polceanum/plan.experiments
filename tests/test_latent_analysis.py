import torch

from latent_kv.latent_analysis import categorize_prompt, pca_2d


def test_categorize_prompt_assigns_money_rate_and_comparison_labels():
    categories, primary, difficulty, notes = categorize_prompt(
        "A store sells 6 boxes each day for $4 per box. After three days, how much more money did it earn than last week?"
    )

    assert "money_price_profit" in categories
    assert "rate_time_work" in categories
    assert "comparison_relative" in categories
    assert primary == "rate_time_work"
    assert difficulty == "multi_step"
    assert notes


def test_categorize_prompt_assigns_linear_fraction_labels():
    categories, primary, difficulty, _ = categorize_prompt(
        "Mia gave away half of her stickers and then had 12 left. How many stickers did she have originally?"
    )

    assert "fractions_ratios_percents" in categories
    assert "linear_equation" in categories
    assert primary == "linear_equation"
    assert difficulty in {"two_step", "multi_step"}


def test_categorize_prompt_treats_a_day_as_rate_language():
    categories, primary, _, _ = categorize_prompt("James delivers 600 newspapers in a day for 5 days.")

    assert "rate_time_work" in categories
    assert primary == "rate_time_work"


def test_pca_2d_returns_finite_coordinates_and_variance():
    latents = torch.tensor(
        [
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 1.0],
            [1.0, 1.0, 1.0],
        ]
    )

    coords, explained = pca_2d(latents)

    assert coords.shape == (4, 2)
    assert torch.isfinite(coords).all()
    assert len(explained) == 2
    assert explained[0] >= 0.0
    assert sum(explained) <= 1.000001
