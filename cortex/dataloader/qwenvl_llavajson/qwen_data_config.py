import re


def _subtask_config(
    annotation_path,
    data_path,
    video_keys,
    sample_interleave,
    transition_tail_sec,
    transition_head_sec,
    dense_sample_step,
    ignore_boundary_sec,
    final_tail_sample_step,
    last_tail_sec,
):
    return {
        "annotation_path": annotation_path,
        "data_path": data_path,
        "video_keys": video_keys,
        "observation_absolute_timestamps_sec": [0.0],
        "observation_time_offsets_sec": [-1.0, -0.5, 0.0],
        "sample_interleave": sample_interleave,
        "transition_tail_sec": transition_tail_sec,
        "transition_head_sec": transition_head_sec,
        "dense_sample_step": dense_sample_step,
        "ignore_boundary_sec": ignore_boundary_sec,
        "final_tail_sample_step": final_tail_sample_step,
        "last_tail_sec": last_tail_sec,
    }


data_dict = {
    "agibot_subtask_train": _subtask_config(
        annotation_path="/path/to/vla_sys2_data/agibot/merged_episodes_norm_lang_mem_train.jsonl",
        data_path="s3://manip-dataset/public/real/agibotworld",
        video_keys="observation.images.head,observation.images.hand_left,observation.images.hand_right",
        sample_interleave=12,
        transition_tail_sec=0.7,
        transition_head_sec=0.5,
        dense_sample_step=6,
        ignore_boundary_sec=0.2,
        final_tail_sample_step=3,
        last_tail_sec=1.0,
    ),
    "agibot_subtask_val": _subtask_config(
        annotation_path="/path/to/vla_sys2_data/agibot/merged_episodes_norm_lang_mem_val.jsonl",
        data_path="s3://manip-dataset/public/real/agibotworld",
        video_keys="observation.images.head,observation.images.hand_left,observation.images.hand_right",
        sample_interleave=12,
        transition_tail_sec=0.7,
        transition_head_sec=0.5,
        dense_sample_step=6,
        ignore_boundary_sec=0.2,
        final_tail_sample_step=3,
        last_tail_sec=1.0,
    ),
    "galaxea_subtask_train": _subtask_config(
        annotation_path="/path/to/vla_sys2_data/galaxea/merged_episodes_norm_lang_mem_train.jsonl",
        data_path="s3://manip-dataset/public/real/galaxea/lerobot_opensource",
        video_keys="observation.images.head_rgb,observation.images.left_wrist_rgb,observation.images.right_wrist_rgb",
        sample_interleave=9,
        transition_tail_sec=1.7,
        transition_head_sec=0.4,
        dense_sample_step=5,
        ignore_boundary_sec=0.1,
        final_tail_sample_step=3,
        last_tail_sec=1.5,
    ),
    "galaxea_subtask_val": _subtask_config(
        annotation_path="/path/to/vla_sys2_data/galaxea/merged_episodes_norm_lang_mem_val.jsonl",
        data_path="s3://manip-dataset/public/real/galaxea/lerobot_opensource",
        video_keys="observation.images.head_rgb,observation.images.left_wrist_rgb,observation.images.right_wrist_rgb",
        sample_interleave=9,
        transition_tail_sec=1.7,
        transition_head_sec=0.4,
        dense_sample_step=5,
        ignore_boundary_sec=0.1,
        final_tail_sample_step=3,
        last_tail_sec=1.5,
    ),
    "behavior_subtask_train": _subtask_config(
        annotation_path="/path/to/vla_sys2_data/behaviro-1k/merged_episodes_norm_lang_mem_train.jsonl",
        data_path="/path/to/behavior-1k/2025-challenge-demos/",
        video_keys="observation.images.rgb.head,observation.images.rgb.left_wrist,observation.images.rgb.right_wrist",
        sample_interleave=30,
        transition_tail_sec=0.7,
        transition_head_sec=1.0,
        dense_sample_step=10,
        ignore_boundary_sec=0.2,
        final_tail_sample_step=3,
        last_tail_sec=1.5,
    ),
    "behavior_subtask_val": _subtask_config(
        annotation_path="/path/to/vla_sys2_data/behaviro-1k/merged_episodes_norm_lang_mem_val.jsonl",
        data_path="/path/to/behavior-1k/2025-challenge-demos/",
        video_keys="observation.images.rgb.head,observation.images.rgb.left_wrist,observation.images.rgb.right_wrist",
        sample_interleave=30,
        transition_tail_sec=0.7,
        transition_head_sec=1.0,
        dense_sample_step=10,
        ignore_boundary_sec=0.2,
        final_tail_sample_step=3,
        last_tail_sec=1.5,
    ),
}


def parse_sampling_rate(dataset_name):
    match = re.search(r"%(\d+)$", dataset_name)
    if match:
        return int(match.group(1)) / 100.0
    return 1.0


def data_list(dataset_names):
    if dataset_names == ["all"]:
        dataset_names = list(data_dict.keys())

    config_list = []
    for dataset_name in dataset_names:
        sampling_rate = parse_sampling_rate(dataset_name)
        dataset_name = re.sub(r"%(\d+)$", "", dataset_name)
        if dataset_name not in data_dict:
            available = ", ".join(sorted(data_dict))
            raise ValueError(f"do not find {dataset_name}; available datasets: {available}")
        config = data_dict[dataset_name].copy()
        config["dataset_name"] = dataset_name
        config["sampling_rate"] = sampling_rate
        config_list.append(config)
    return config_list
