"""Train with a method's pipeline steps attached.

Upstream's ``train()`` is a single 400-line function carrying accelerate, FSDP,
resume and logging. Reimplementing it to insert one processor step would mean owning
all of that; instead this attaches at the only seam that matters -- the point where
the preprocessor pipeline is built -- and calls upstream's ``train()`` unchanged.

The config is upstream's ``TrainPipelineConfig`` plus one field, so every existing
lerobot-train flag keeps working and ``--method.type`` composes with them:

    python -m robot_stack.train.run_train \\
        --dataset.repo_id=HuggingFaceVLA/libero --policy.type=act \\
        --method.type=demospeedup --method.labels_path=outputs/labels \\
        --output_dir=outputs/train/speedup
"""

from dataclasses import dataclass, field

from lerobot.configs import parser
from lerobot.configs.train import TrainPipelineConfig
from lerobot.scripts import lerobot_train

from robot_stack.methods.config import MethodConfig, NoMethod


@dataclass
class SpeedupTrainConfig(TrainPipelineConfig):
    """Upstream's training config, plus the method choice."""

    method: MethodConfig = field(default_factory=NoMethod)


def attach_method_steps(method: MethodConfig):
    """Wrap the processor factory so the method's steps join the pipeline.

    Preprocessor steps are inserted *before the normalizer*: DemoSpeedup substitutes
    each sample's action chunk from the raw episode table, so it must run on raw
    actions. Postprocessor steps are appended, which is where PACE would land if a
    method contributed both.

    Patching the name inside ``lerobot_train`` rather than at its source keeps the
    effect scoped to this training run -- nothing else in the process sees a changed
    factory.
    """
    original_factory = lerobot_train.make_pre_post_processors
    original_make_datasets = lerobot_train.make_train_eval_datasets
    captured: dict = {}

    def make_datasets(cfg, *args, **kwargs):
        # The processor factory is not handed the dataset, but DemoSpeedup preloads
        # episode action trajectories from it -- captured here. The chunk halving
        # also happens at this point, before the policy is built.
        datasets = original_make_datasets(cfg, *args, **kwargs)
        captured["dataset"] = datasets[0]
        method.adjust_policy_after_datasets(cfg.policy)
        return datasets

    def patched(*args, **kwargs):
        preprocessor, postprocessor = original_factory(*args, **kwargs)
        # The retiming step substitutes raw episode actions, so it must see the
        # batch before normalization: insert ahead of the normalizer rather than
        # appending. (Index selection commutes with per-dim affine normalization,
        # so this matches upstream's normalize-then-retime semantics.)
        steps = method.preprocessor_steps(captured.get("dataset"))
        at = next(
            (i for i, step in enumerate(preprocessor.steps) if "Normalizer" in type(step).__name__),
            len(preprocessor.steps),
        )
        preprocessor.steps[at:at] = steps
        postprocessor.steps.extend(method.postprocessor_steps())
        return preprocessor, postprocessor

    lerobot_train.make_train_eval_datasets = make_datasets
    lerobot_train.make_pre_post_processors = patched
    return original_factory


@parser.wrap()
def main(cfg: SpeedupTrainConfig) -> None:
    attach_method_steps(cfg.method)

    # Call the *undecorated* train(). Upstream's train() is itself @parser.wrap()-ed,
    # and that wrapper only accepts an already-parsed config when
    # `type(args[0]) is TrainPipelineConfig` -- an identity check, which a subclass
    # fails. Handed a SpeedupTrainConfig it silently re-parses sys.argv as a plain
    # TrainPipelineConfig, and every --method.* argument becomes "unrecognized".
    lerobot_train.train.__wrapped__(cfg)


if __name__ == "__main__":
    # Import the package-qualified `main` rather than calling the one defined above.
    # Under `python -m`, this file is executed as `__main__`, so SpeedupTrainConfig
    # would be a *different* class object from the one draccus resolves through the
    # package -- and the `--method.*` arguments silently vanish from the parser.
    from robot_stack.train.run_train import main as packaged_main

    packaged_main()
