from src.fix_checkpoint import should_copy_checkpoint_metadata


def test_fix_checkpoint_copies_only_model_metadata():
    allowed = [
        "config.json",
        "generation_config.json",
        "preprocessor_config.json",
        "video_preprocessor_config.json",
        "configuration_qwen.py",
    ]
    rejected = [
        "model.safetensors",
        "model.safetensors-00001-of-00001.safetensors",
        "model.safetensors.index.json",
        "tokenizer.json",
        "README.md",
        ".gitattributes",
    ]

    assert all(should_copy_checkpoint_metadata(name) for name in allowed)
    assert not any(should_copy_checkpoint_metadata(name) for name in rejected)
