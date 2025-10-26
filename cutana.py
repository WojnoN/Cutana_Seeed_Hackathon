# all_policies.py
# Roll out multiple pretrained policies back-to-back without saving any dataset.
# No dataset creation, no eval_* suffix checks, no video/image writers—pure control.

import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from pprint import pformat
from typing import Any, List, Optional

from lerobot.cameras import CameraConfig  # noqa: F401
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig  # noqa: F401
from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig  # noqa: F401
from lerobot.configs import parser
from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.pipeline_features import (
    aggregate_pipeline_dataset_features,
    create_initial_features,
)
from lerobot.datasets.utils import combine_feature_dicts
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.utils import make_robot_action
from lerobot.processor import (
    PolicyAction,
    PolicyProcessorPipeline,
    RobotAction,
    RobotObservation,
    RobotProcessorPipeline,
    make_default_processors,
)
from lerobot.processor.rename_processor import rename_stats
from lerobot.robots import (  # noqa: F401
    Robot,
    RobotConfig,
    bi_so100_follower,
    hope_jr,
    koch_follower,
    make_robot_from_config,
    so100_follower,
    so101_follower,
)
from lerobot.teleoperators import (  # noqa: F401
    Teleoperator,
    TeleoperatorConfig,
    bi_so100_leader,
    homunculus,
    koch_leader,
    make_teleoperator_from_config,
    so100_leader,
    so101_leader,
)
from lerobot.teleoperators.keyboard.teleop_keyboard import KeyboardTeleop
from lerobot.utils.constants import ACTION, OBS_STR
from lerobot.utils.control_utils import (
    init_keyboard_listener,
    is_headless,
    predict_action,
)
from lerobot.utils.import_utils import register_third_party_devices
from lerobot.utils.utils import (
    get_safe_torch_device,
    init_logging,
    log_say,
)
from lerobot.utils.robot_utils import busy_wait
from lerobot.utils.visualization_utils import init_rerun, log_rerun_data

from lerobot.datasets.lerobot_dataset import LeRobotDataset


# ────────────────────────────────────────────────────────────────────────────────
# Config dataclasses
# ────────────────────────────────────────────────────────────────────────────────

# @dataclass
# class DatasetReplayConfig:
#     # Dataset identifier. By convention it should match '{hf_username}/{dataset_name}' (e.g. `lerobot/test`).
#     repo_id: str
#     # Episode to replay.
#     episode: int
#     # Root directory where the dataset will be stored (e.g. 'dataset/path').
#     root: str | Path | None = None
#     # Limit the frames per second. By default, uses the policy fps.
#     fps: int = 30

# @dataclass
# class ReplayConfig:
#     robot: RobotConfig
#     dataset: DatasetReplayConfig
#     # Use vocal synthesis to read events.
#     play_sounds: bool = True

def replay_dataset_last_episode(
    *,
    robot: Robot,
    events: dict,
    dataset_id: str,
    fps: int,
    robot_action_processor: RobotProcessorPipeline[tuple[RobotAction, RobotObservation], RobotAction],
    robot_observation_processor: RobotProcessorPipeline[RobotObservation, RobotObservation],
    root: str | Path | None = None,
) -> None:
    """
    Read-only replay: streams the LAST episode's actions from a dataset to the robot at `fps`.
    No dataset creation or writing occurs.
    """
    ds = LeRobotDataset(dataset_id, root=root, batch_encoding_size=1)
    total_eps = ds.num_episodes
    if total_eps == 0:
        logging.warning(f"[replay] No episodes in dataset: {dataset_id}")
        return

    ep_index = total_eps - 1
    logging.info(f"[replay] {dataset_id}: playing episode_index={ep_index}")

    # Pull only that episode's frames
    ds_slice = ds.hf_dataset.filter(lambda x: x["episode_index"] == ep_index)
    if len(ds_slice) == 0:
        logging.warning(f"[replay] Empty slice for episode_index={ep_index} in {dataset_id}")
        return

    actions = ds_slice.select_columns(ACTION)
    action_names = ds.features[ACTION]["names"]

    for i in range(len(ds_slice)):
        if events.get("stop_recording"):
            break

        t0 = time.perf_counter()
        arr = actions[i][ACTION]
        # Rebuild the action dict
        action = {name: arr[j] for j, name in enumerate(action_names)}

        # Allow your processors to run with the current observation
        obs = robot.get_observation()
        obs_proc = robot_observation_processor(obs)
        act_proc = robot_action_processor((action, obs_proc))

        _ = robot.send_action(act_proc)

        # Pace to requested fps
        dt = time.perf_counter() - t0
        busy_wait(1.0 / fps - dt)


class _RolloutMeta:
    """
    Minimal dataset-like metadata to satisfy policies.factory.make_policy.
    We only include what many policies reasonably expect: fps, features, and stats.
    """
    def __init__(self, *, fps: int, features: dict, stats: dict | None = None, robot_type: str | None = None):
        self.fps = fps
        self.features = features
        self.stats = stats or {}
        self.robot_type = robot_type
        # nice-to-have fields some code may log:
        self.repo_id = "rollout-only"
        self.num_episodes = 0
        self.version = "rollout-only"


@dataclass
class DatasetRecordConfig:
    # We keep these for CLI parity, but they are NOT used to write anything in rollout-only mode.
    repo_id: str
    single_task: str
    root: str | Path | None = None
    fps: int = 30
    episode_time_s: int | float = 30
    reset_time_s: int | float = 0
    num_episodes: int = 1
    video: bool = True  # ignored in rollout-only mode
    push_to_hub: bool = False  # ignored
    private: bool = False  # ignored
    tags: list[str] | None = None  # ignored
    num_image_writer_processes: int = 1  # ignored
    num_image_writer_threads_per_camera: int = 8  # ignored
    video_encoding_batch_size: int = 1  # ignored
    rename_map: dict[str, str] = field(default_factory=dict)


@dataclass
class RecordConfig:
    robot: RobotConfig
    dataset: DatasetRecordConfig

    # Teleop optional (you can mix teleop/policy, but in this file we focus on policy-only rollout)
    teleop: TeleoperatorConfig | None = None

    # Single policy (optional) — but we mainly use policy_paths below for multi-policy
    policy: PreTrainedConfig | None = None

    # Multi-policy rollout list, e.g. '["user/policyA","user/policyB"]'
    policy_paths: Optional[List[str]] = None

    # Display windows / rerun stream
    display_data: bool = False

    # Voice prompts
    play_sounds: bool = True

    # Resume (ignored in rollout-only; present just to keep CLI compatible)
    resume: bool = False

    # Optional: after each policy, pause this many seconds
    inter_policy_pause_s: float = 0.0


# ────────────────────────────────────────────────────────────────────────────────
# Rollout loop (NO dataset writes)
# ────────────────────────────────────────────────────────────────────────────────

def record_loop_rollout_only(
    *,
    robot: Robot,
    events: dict,
    fps: int,
    teleop_action_processor: RobotProcessorPipeline[
        tuple[RobotAction, RobotObservation], RobotAction
    ],
    robot_action_processor: RobotProcessorPipeline[
        tuple[RobotAction, RobotObservation], RobotAction
    ],
    robot_observation_processor: RobotProcessorPipeline[
        RobotObservation, RobotObservation
    ],
    # policy path
    policy: PreTrainedPolicy | None = None,
    preprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]] | None = None,
    postprocessor: PolicyProcessorPipeline[PolicyAction, PolicyAction] | None = None,
    # synthetic features used to format observation/actions (since there is no dataset)
    features_for_format: dict | None = None,
    # general control
    control_time_s: int | None = None,
    single_task: str | None = None,
    display_data: bool = False,
):
    """
    Rollout loop that:
      - polls robot observation
      - optionally feeds a policy to get an action
      - runs processors
      - sends action to robot
    It does *NOT* write (or even instantiate) any dataset.
    """
    if control_time_s is None:
        raise ValueError("control_time_s must be provided.")

    if policy is not None and (preprocessor is None or postprocessor is None):
        raise ValueError("If a policy is provided, both preprocessor and postprocessor must be provided.")

    if features_for_format is None:
        raise ValueError("features_for_format must be provided in rollout-only mode.")

    # Reset policy and processors if they are provided
    if policy is not None and preprocessor is not None and postprocessor is not None:
        policy.reset()
        preprocessor.reset()
        postprocessor.reset()

    timestamp = 0.0
    start_episode_t = time.perf_counter()

    # Teleop validation for potential future extension:
    teleop_keyboard = teleop_arm = None

    while timestamp < control_time_s:
        loop_t0 = time.perf_counter()

        if events.get("exit_early"):
            events["exit_early"] = False
            break

        # 1) Get observation, then process it
        obs = robot.get_observation()
        obs_processed = robot_observation_processor(obs)

        # Build the dataset-like observation frame (w/o having a dataset)
        # This ensures the downstream policy preprocessor sees the expected keys/shapes.
        from lerobot.datasets.utils import build_dataset_frame  # imported here to emphasize no dataset creation
        observation_frame = build_dataset_frame(features_for_format, obs_processed, prefix=OBS_STR)

        # 2) Get action (policy only in this rollout-only mode)
        if policy is not None:
            action_values = predict_action(
                observation=observation_frame,
                policy=policy,
                device=get_safe_torch_device(policy.config.device),
                preprocessor=preprocessor,
                postprocessor=postprocessor,
                use_amp=policy.config.use_amp,
                task=single_task,
                robot_type=robot.robot_type,
            )
            act_processed_policy: RobotAction = make_robot_action(action_values, features_for_format)
            robot_action_to_send = robot_action_processor((act_processed_policy, obs))
            if display_data:
                log_rerun_data(observation=obs_processed, action=act_processed_policy)
        else:
            # No policy provided: nothing to send (extend as-needed for teleop)
            logging.info("No policy provided; skipping action. (This file is rollout-only for policies.)")
            break

        # 3) Send action to robot
        _ = robot.send_action(robot_action_to_send)

        # 4) Maintain control rate
        dt_s = time.perf_counter() - loop_t0
        busy_wait(1.0 / fps - dt_s)

        timestamp = time.perf_counter() - start_episode_t


# ────────────────────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────────────────────

def _load_policy_bundle_for_rollout(
    policy_path: str,
    *,
    rename_map: dict[str, str],
    features_for_format: dict,
    fps: int,
    robot_type: str | None,
) -> tuple[PreTrainedConfig, PreTrainedPolicy, PolicyProcessorPipeline, PolicyProcessorPipeline]:
    """
    Loads:
      - PreTrainedConfig (draccus-aware)
      - Policy instance (with a minimal ds_meta to satisfy factory)
      - Pre/Post processors (fed empty stats + rename_map)
    """
    cli_overrides = parser.get_cli_overrides("policy")
    cfg = PreTrainedConfig.from_pretrained(policy_path, cli_overrides=cli_overrides)
    cfg.pretrained_path = policy_path

    # Minimal in-memory meta to satisfy make_policy’s requirement.
    meta = _RolloutMeta(
        fps=fps,
        features=features_for_format,
        stats={},                      # keep empty; policies with built-in norms will still work
        robot_type=robot_type,
    )

    # Create policy with meta (no files are written—this is just to satisfy the factory)
    policy = make_policy(cfg, ds_meta=meta)

    # Build pre/post processors with empty stats + rename overrides
    preproc, postproc = make_pre_post_processors(
        policy_cfg=cfg,
        pretrained_path=cfg.pretrained_path,
        dataset_stats=rename_stats({}, rename_map),
        preprocessor_overrides={
            "device_processor": {"device": cfg.device},
            "rename_observations_processor": {"rename_map": rename_map},
        },
    )
    return cfg, policy, preproc, postproc



def _compute_features_for_rollout(
    *,
    teleop_action_processor,
    robot_observation_processor,
    robot: Robot,
    use_videos: bool,
) -> dict:
    """
    Compute a *synthetic* "dataset_features" dict entirely in-memory, based on the processors and the robot.
    This is only used to format observations/actions consistently—no files are written.
    """
    # Action side comes from robot.action_features (no teleop in this file)
    action_features = create_initial_features(action=robot.action_features)

    # Observation side comes from robot.observation_features
    obs_features = create_initial_features(observation=robot.observation_features)

    features = combine_feature_dicts(
        aggregate_pipeline_dataset_features(
            pipeline=teleop_action_processor,
            initial_features=action_features,
            use_videos=use_videos,
        ),
        aggregate_pipeline_dataset_features(
            pipeline=robot_observation_processor,
            initial_features=obs_features,
            use_videos=use_videos,
        ),
    )
    return features


# ────────────────────────────────────────────────────────────────────────────────
# Entry point
# ────────────────────────────────────────────────────────────────────────────────

@parser.wrap()
def record(cfg: RecordConfig) -> None:
    init_logging()
    logging.info(pformat(asdict(cfg)))

    if cfg.display_data:
        init_rerun(session_name="rollout")

    # Robot & (optional) teleop
    robot = make_robot_from_config(cfg.robot)
    teleop = make_teleoperator_from_config(cfg.teleop) if cfg.teleop is not None else None

    teleop_action_processor, robot_action_processor, robot_observation_processor = make_default_processors()

    # Compute synthetic features for formatting frames (no dataset object needed)
    features_for_format = _compute_features_for_rollout(
        teleop_action_processor=teleop_action_processor,
        robot_observation_processor=robot_observation_processor,
        robot=robot,
        use_videos=cfg.dataset.video,
    )

    robot.connect()
    if teleop is not None:
        teleop.connect()

    listener, events = init_keyboard_listener()

    try:
        # ── Multi-policy rollout (no dataset writes) ─────────────────────────────
        policy_paths = cfg.policy_paths or []
        if not policy_paths and cfg.policy is not None and cfg.policy.pretrained_path is not None:
            # Allow single policy via --policy too
            policy_paths = [str(cfg.policy.pretrained_path)]
        if not policy_paths:
            raise ValueError(
                "No policies provided. Use --policy_paths='[\"user/policyA\",\"user/policyB\"]' "
                "or --policy.path=<repo_or_dir> via the --policy.* CLI."
            )

        for i, policy_path in enumerate(policy_paths, start=1):
            log_say(f"Loading policy {i}/{len(policy_paths)}: {policy_path}", cfg.play_sounds)
            tmp_cfg, policy, preproc, postproc = _load_policy_bundle_for_rollout(
                policy_path,
                rename_map=cfg.dataset.rename_map,
                features_for_format=features_for_format,
                fps=cfg.dataset.fps,
                robot_type=getattr(robot, "name", None),
            )


            # Episode loop for this policy
            for ep in range(cfg.dataset.num_episodes):
                log_say(f"Policy [{policy_path}] — Episode {ep+1}/{cfg.dataset.num_episodes}", cfg.play_sounds)
                record_loop_rollout_only(
                    robot=robot,
                    events=events,
                    fps=cfg.dataset.fps,
                    teleop_action_processor=teleop_action_processor,
                    robot_action_processor=robot_action_processor,
                    robot_observation_processor=robot_observation_processor,
                    policy=policy,
                    preprocessor=preproc,
                    postprocessor=postproc,
                    features_for_format=features_for_format,
                    control_time_s=cfg.dataset.episode_time_s,
                    single_task=cfg.dataset.single_task,
                    display_data=cfg.display_data,
                )
                if events.get("stop_recording"):
                    break

                # Optional reset wait between episodes (pure sleep; feel free to replace with robot reset)
                if cfg.dataset.reset_time_s and cfg.dataset.reset_time_s > 0:
                    t0 = time.perf_counter()
                    while time.perf_counter() - t0 < cfg.dataset.reset_time_s and not events.get("stop_recording"):
                        busy_wait(0.01)

            if events.get("stop_recording"):
                break

            # Optional pause between policies
            if cfg.inter_policy_pause_s and cfg.inter_policy_pause_s > 0.0:
                t0 = time.perf_counter()
                while time.perf_counter() - t0 < cfg.inter_policy_pause_s and not events.get("stop_recording"):
                    busy_wait(0.01)

        log_say("Homing the arm.", cfg.play_sounds)
        try:
            replay_dataset_last_episode(
                robot=robot,
                events=events,
                dataset_id="vednot25t/last_stop1",      # if your dataset is on the Hub under a namespace, use "owner/last_stop1"
                fps=cfg.dataset.fps,          # or use the dataset's own fps if you prefer
                robot_action_processor=robot_action_processor,
                robot_observation_processor=robot_observation_processor,
                root=None,                    # set a path here if it's a local dataset folder
            )
        except Exception as e:
            logging.exception(f"Failed to replay 'last_stop1': {e}")

    finally:
        # Clean up
        robot.disconnect()
        if teleop is not None:
            teleop.disconnect()
        if not is_headless() and listener is not None:
            listener.stop()

        log_say("Exiting", cfg.play_sounds)


def main():
    register_third_party_devices()
    record()


if __name__ == "__main__":
    main()
