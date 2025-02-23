import json
import multiprocessing
import os
import platform

import wandb
import yaml

from etalon.capacity_search.capacity_search import CapacitySearch
from etalon.capacity_search.config.config import JobConfig
from etalon.capacity_search.main import get_parser
from etalon.logger import init_logger

logger = init_logger(__name__)


def num_fixed_qps_values(capacity_search: CapacitySearch):
    """Wrapper function to get the number of fixed QPS values."""
    # it makes use of the capacity_search object to get the number of fixed QPS values

    logger.info(
        f"Starting runs for {capacity_search.job_config.get_human_readable_name()}",
    )

    tbt_at_max_qps = None
    ttft_at_max_qps = None
    tpot_at_max_qps = None
    deadline_miss_rate_at_max_qps = None
    best_run_id = None
    found_valid_qps = False

    max_qps_under_sla = None
    min_qps_over_sla = 2**32

    logger.info(f"Fixed QPS values: {capacity_search.args.fixed_qps_values}")
    for qps in capacity_search.args.fixed_qps_values:
        logger.info(f"Running for QPS: {qps}")
        (
            is_under_sla,
            tbt,
            ttft,
            tpot,
            deadline_miss_rate,
            run_id,
        ) = capacity_search.is_under_sla(qps)

        if is_under_sla:
            found_valid_qps = True
            max_qps_under_sla = qps
            tbt_at_max_qps = tbt
            ttft_at_max_qps = ttft
            tpot_at_max_qps = tpot
            deadline_miss_rate_at_max_qps = deadline_miss_rate
            best_run_id = run_id
        else:
            min_qps_over_sla = min(min_qps_over_sla, qps)

    if not found_valid_qps:
        logger.info(
            f"No valid QPS found for {capacity_search.job_config.get_human_readable_name()}",
        )
        return {}

    logger.info(
        f"Max QPS under SLO for {capacity_search.job_config.get_human_readable_name()} - "
        f"QPS: {max_qps_under_sla}, "
        f"TBT P{capacity_search.args.tbt_percentile * 100}: {tbt_at_max_qps}, "
        f"TTFT P{capacity_search.args.ttft_percentile * 100}: {ttft_at_max_qps}, "
        f"TPOT P{capacity_search.args.tpot_percentile * 100}: {tpot_at_max_qps}, "
        f"Deadline Miss Rate P{capacity_search.args.deadline_miss_rate_percentile * 100}: {deadline_miss_rate_at_max_qps}"
        f"Best Run ID: {best_run_id}",
    )

    if (
        capacity_search.args.wandb_project is not None
        and capacity_search.args.enable_wandb_sweep
    ):
        best_run = wandb.Api().run(
            f"{capacity_search.args.wandb_project}/{best_run_id}"
        )
        best_run.tags.append("BEST_CONFIG")
        best_run.update()

    return {
        **capacity_search.job_config.to_config_dict(),
        "max_qps_under_sla": max_qps_under_sla,
        "deadline_miss_rate_at_max_qps": deadline_miss_rate_at_max_qps,
    }


def setup():
    """Setup function to parse the arguments and setup the config."""

    parser = get_parser()

    parser.add_argument(
        "--fixed-qps-values",
        type=float,
        nargs="+",
        default=[0.25, 0.75, 1.25, 1.75],
        help="Fixed QPS values to search for",
    )

    args = parser.parse_args()

    if args.wandb_project and args.enable_wandb_sweep:
        assert (
            args.wandb_sweep_name or args.wandb_sweep_id
        ), "wandb-sweep-name/id is required with wandb-project"

    config = yaml.safe_load(open(args.config_path))
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


if __name__ == "__main__":
    if platform.system() == "Darwin":
        multiprocessing.set_start_method("fork", force=True)

    args, config = setup()
    job_configs = JobConfig.generate_job_configs(config)

    all_results = []

    for job_config in job_configs:
        capacity_search = CapacitySearch(
            job_config,
            args,
        )
        result = num_fixed_qps_values(capacity_search)
        all_results.append(result)
