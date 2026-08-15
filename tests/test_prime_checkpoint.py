from pathlib import Path

import torch

from src.rl.prompt_calibration import PromptCalibrationHead
from src.train_prime import PrimeGRPOTrainer


def test_prm_artifacts_round_trip_model_optimizer_calibration_and_step(tmp_path):
    trainer = object.__new__(PrimeGRPOTrainer)
    trainer.prm_model = torch.nn.Linear(3, 2)
    trainer.prompt_calibration_head = PromptCalibrationHead(3)
    parameters = list(trainer.prm_model.parameters()) + list(
        trainer.prompt_calibration_head.parameters()
    )
    trainer.prm_optimizer = torch.optim.AdamW(parameters, lr=1e-3)
    trainer._prime_step = 17

    loss = trainer.prm_model(torch.ones(1, 3)).sum()
    loss = loss + trainer.prompt_calibration_head(torch.ones(1, 3)).sum()
    loss.backward()
    trainer.prm_optimizer.step()

    expected_model = {
        name: value.detach().clone() for name, value in trainer.prm_model.state_dict().items()
    }
    expected_head = {
        name: value.detach().clone()
        for name, value in trainer.prompt_calibration_head.state_dict().items()
    }
    trainer._save_prm_artifacts(tmp_path)

    with torch.no_grad():
        for parameter in trainer.prm_model.parameters():
            parameter.zero_()
        for parameter in trainer.prompt_calibration_head.parameters():
            parameter.zero_()
    trainer._prime_step = 0
    trainer.prm_optimizer.state.clear()

    trainer._load_prm_checkpoint(tmp_path)

    assert trainer._prime_step == 17
    assert (Path(tmp_path) / "prm" / "model.pt").is_file()
    assert (Path(tmp_path) / "prm" / "optimizer.pt").is_file()
    assert (Path(tmp_path) / "prm" / "state.pt").is_file()
    for name, value in trainer.prm_model.state_dict().items():
        assert torch.equal(value, expected_model[name])
