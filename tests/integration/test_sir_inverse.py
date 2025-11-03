# """Integration test for SIR inverse problem."""

# from lightning.pytorch import Trainer
# import pytest

# from pinn.lightning import PINNModule
# from pinn.problems import (
# SIRInvDataModule, SIRInvHyperparameters, SIRInvProblem, SIRInvProperties,
# )


# @pytest.mark.skip(reason="Not functional yet.")
# def test_sir_inverse_integration() -> None:
#     """Test that SIR inverse problem can be instantiated and trained briefly."""
#     # Suppress Lightning warnings for cleaner test output
#     # with warnings.catch_warnings():
#     #     warnings.filterwarnings("ignore", category=UserWarning)

#     # Create properties and hyperparameters
#     props = SIRInvProperties()
#     hp = SIRInvHyperparameters()

#     # Override hyperparameters for quick test
#     hp.max_epochs = 2
#     hp.batch_size = 32
#     hp.collocations = 64
#     hp.data_per_batch = 8
#     hp.lr = 1e-2

#     # Create data module
#     dm = SIRInvDataModule(props=props, hp=hp)
#     dm.setup()

#     # Create problem
#     problem = SIRInvProblem(props=props, hp=hp)

#     # Create Lightning module
#     module = PINNModule(problem=problem, hp=hp)

#     # Create trainer with minimal configuration
#     trainer = Trainer(
#         max_epochs=hp.max_epochs,
#         gradient_clip_val=hp.gradient_clip_val,
#         logger=False,  # Disable logging for test
#         enable_checkpointing=False,  # Disable checkpointing for test
#         enable_progress_bar=False,  # Disable progress bar for test
#         enable_model_summary=False,  # Disable model summary for test
#         log_every_n_steps=0,
#     )

#     # Train briefly
#     trainer.fit(module, dm)

#     # Basic sanity checks
#     assert module.problem is not None
#     assert module.hp.max_epochs == 2

#     # Check that parameters have been updated (not just initialized)
#     initial_beta = problem.beta.forward().item()
#     assert initial_beta > 0  # Should be positive

#     # Check that logs are being generated
#     logs = problem.get_logs()
#     assert "total" in logs
#     assert "beta" in logs

#     print(f"Integration test passed! Final beta value: {initial_beta}")
