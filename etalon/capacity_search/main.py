import argparse
import json
import multiprocessing
import os
import platform
import time

import wandb
import yaml

from etalon.capacity_search.search_manager import SearchManager
from etalon.logger import init_logger

logger = init_logger(__name__)


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--min-search-granularity",
        type=float,
        default=2.5,
        help="Minimum search granularity for capacity (%)",
    )
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument(
        "--profile-dir",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--config-path",
        type=str,
        default="./etalon/capacity_search/config/default_config.yml",
    )
    parser.add_argument("--slo-type", type=str, default="deadline")
    parser.add_argument("--tbt-slo", type=float, default=0.03)
    parser.add_argument("--tbt-percentile", type=float, default=0.99)
    parser.add_argument("--ttft-slo", type=float, default=0.1)
    parser.add_argument("--ttft-percentile", type=float, default=0.9)
    parser.add_argument("--tpot-slo", type=float, default=0.1)
    parser.add_argument("--tpot-percentile", type=float, default=0.9)
    parser.add_argument("--ttft-slack-slo", type=float, default=0.3)
    parser.add_argument("--deadline-miss-rate-slo", type=float, default=0.1)
    parser.add_argument("--deadline-miss-rate-percentile", type=float, default=0.99)
    parser.add_argument(
        "--dynamic-ttft-slo",
        type=bool,
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--max-iterations", type=int, default=20)
    parser.add_argument(
        "--time-limit", type=int, default=20, help="Time limit in minutes"
    )
    parser.add_argument(
        "--debug", action="store_true", help="Print debug logs and commands"
    )
    parser.add_argument("--wandb-project", type=str, default=None)
    parser.add_argument("--wandb-group", type=str, default=None)
    parser.add_argument(
        "--should-write-metrics-to-wandb",
        type=bool,
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--wandb-sweep-name", type=str, default=None)
    parser.add_argument("--wandb-sweep-id", type=str, default=None)
    parser.add_argument(
        "--enable-wandb-sweep",
        type=bool,
        action=argparse.BooleanOptionalAction,
        default=False,
    )

    return parser


def setup():
    """Setup function to parse the arguments and setup the config."""

    parser = get_parser()
    args = parser.parse_args()

    if args.wandb_project and args.enable_wandb_sweep:
        assert (
            args.wandb_sweep_name or args.wandb_sweep_id
        ), "wandb-sweep-name/id is required with wandb-project"

    config = yaml.safe_load(open(args.config_path))

    assert args.deadline_miss_rate_slo >= 0 and args.deadline_miss_rate_slo <= 1

    os.makedirs(args.output_dir, exist_ok=True)

    # merge the config with the args
    config.update(vars(args))
    logger.info(f"Config: {config}")

    # store the config and args
    json.dump(config, open(f"{args.output_dir}/config.json", "w"))

    if args.wandb_project and args.enable_wandb_sweep and not args.wandb_sweep_id:
        config["name"] = args.wandb_sweep_name
        config["method"] = "custom"

        sweep_id = wandb.sweep(config, project=args.wandb_project)
        args.wandb_sweep_id = sweep_id
        # required so that wandb doesn't delay flush of child logs
        wandb.finish(quiet=True)

    return args, config


def run():
    logger.info("Starting capacity search")
    args, config = setup()
    search_manager = SearchManager(args, config)
    start_time = time.time()
    all_results = search_manager.run()
    end_time = time.time()
    logger.info(f"Benchmarking took time: {end_time - start_time}")


if __name__ == "__main__":
    if platform.system() == "Darwin":
        multiprocessing.set_start_method("fork", force=True)

    run()
