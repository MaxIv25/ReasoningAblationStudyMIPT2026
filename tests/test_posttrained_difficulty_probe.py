from scripts.probe_posttrained_difficulty import select_stratified_probe, summarize


def test_probe_selection_is_balanced_and_reproducible():
    records = [
        {"example_id": f"gsm8k:{index}", "source": "gsm8k"}
        for index in range(20)
    ]
    subjects = ("algebra", "geometry")
    for level in (1, 2):
        for subject in subjects:
            records.extend(
                {
                    "example_id": f"math:{level}:{subject}:{index}",
                    "source": "math",
                    "level": level,
                    "subject": subject,
                }
                for index in range(10)
            )

    selected = select_stratified_probe(
        records,
        gsm8k_count=6,
        math_levels=(1, 2),
        math_subjects=subjects,
        math_per_subject=3,
        seed=42,
    )

    assert selected == select_stratified_probe(
        records,
        gsm8k_count=6,
        math_levels=(1, 2),
        math_subjects=subjects,
        math_per_subject=3,
        seed=42,
    )
    assert len(selected) == 18
    assert sum(item["source"] == "gsm8k" for item in selected) == 6
    for level in (1, 2):
        for subject in subjects:
            assert sum(
                item.get("level") == level and item.get("subject") == subject
                for item in selected
            ) == 3
    assert len({item["example_id"] for item in selected}) == len(selected)


def test_probe_summary_reports_grpo_and_prime_acceptance_for_k8():
    records = []
    for index, num_correct in enumerate((0, 1, 2, 6, 7, 8)):
        records.append(
            {
                "example_id": f"gsm8k:{index}",
                "source": "gsm8k",
                "num_correct": num_correct,
                "num_generations": 8,
                "prompt_tokens": 100,
                "completion_token_lengths": [20] * 8,
                "finish_reasons": ["stop"] * 8,
            }
        )

    overall = summarize(records, num_generations=8, seed=42)["overall"]

    assert overall["accuracy"] == 0.5
    assert overall["grpo_informative_rate"] == 4 / 6
    assert overall["prime_accepted_rate"] == 2 / 6
    assert overall["all_wrong_rate"] == 1 / 6
    assert overall["all_correct_rate"] == 1 / 6
