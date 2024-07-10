<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/_static/logo/dark.png">
    <img alt="vLLM" src="docs/_static/logo/light.png" width=50%>
  </picture>
</p>

<h3 align="center">
Tool to benchmark LLM Inference Systems
</h3>

<p align="center">
| <a href="https://project-metron.readthedocs.io/en/latest/"><b>Documentation</b></a> | <a href=""><b>Paper</b></a> |

</p>

---

## Setup

### Clone repository
```bash
git clone https://github.com/project-metron/metron.git
```

### Create conda environment
```bash
conda create -n metron python=3.10
conda activate metron
```

### Install Metron
```bash
cd metron
pip install -e .
```

#### Installing with vLLM (optional)
```bash
pip install -e ".[vllm]"
```

### Setup Wandb [Optional]
First create and setup your account at `https://<your-org>.wandb.io/` or public Wandb and obtain API key. Then run the following command and enter API key linked to your wandb account:
```bash
wandb login --host https://<your-org>.wandb.io
```
To opt out of wandb, do any of the following:
1. Don't pass any wandb related args like `--wandb-project`, `--wandb-group` and `wandb-run-name` when running python scripts. Alternatively, pass in `--no-should-write-metrics` instead of `--should-write-metrics` boolean flag.
2. Run `export WANDB_MODE=disabled` in your shell or add this to `~/.zshrc` or `~/.bashrc`. Remember to reload your shell using `source ~/.zshrc` or `source ~/.bashrc`.

## Running Code

### Running with Public APIs
#### Export API Key and URL
```bash
export OPENAI_API_KEY=secret_abcdefg
export OPENAI_API_BASE=https://api.endpoints.anyscale.com/v1
```
#### Running Benchmark
```bash
python -m metron.run_benchmark \
--model "meta-llama/Meta-Llama-3-8B-Instruct" \
--max-num-completed-requests 150 \
--timeout 600 \
--num-ray-clients 2 \
--num-concurrent-requests-per-client 5 \
--output-dir "result_outputs" \
--request-interval-generator-provider "poisson" \
--poisson-request-interval-generator-qps 0.5 \
--request-length-generator-provider "trace" \
--trace-request-length-generator-trace-file "./data/processed_traces/arxiv_summarization_filtered_stats_llama2_tokenizer.csv" \
--request-generator-max-tokens 8192 \
--ttft-deadline 0.3 \
--tbt-deadline 0.03 \
--should-write-metrics \
--wandb-project Project \
--wandb-group Group \
--wandb-run-name Run
```

There are many more arguments for running benchmark, run the following to know more:
```bash
python -m metron.run_benchmark -h
```

### Running with Open Source Systems
Metron can be run with any open source LLM inference system. If open source system does not provide OpenAI Compatible APIs, then kindly implement new LLM clients to support new open source system as explained in [here](#implementing-new-llm-clients).

Here we give an example with vLLM.

#### Launch vLLM Server
```bash
python -m vllm.entrypoints.openai.api_server --model meta-llama/Meta-Llama-3-8B-Instruct --dtype auto --api-key token-abc123 -tp 1 --rope-scaling '{"type":"dynamic","factor":2.0}'
```

If we need higher context length than supported by the model with certain scale factor, then we can add rope-scaling as `--rope-scaling '{"type":"dynamic","factor":2.0}'`. Adjust type and factor as per the use case.

#### Export API Key and URL
```bash
export OPENAI_API_KEY=token-abc123
export OPENAI_API_BASE=http://localhost:8000/v1
```

And then we can run the benchmark as shown [here](#running-benchmark). Be sure to update `--model` flag to same model used to launch vLLM.

### Saving Results

The results of the benchmark are saved in the results directory specified by the `--output-dir` argument.

## Running Prefill Profiler
To profile prefill times of open source systems and create a prefill time predictor for a given model and open source system, based on input prompt length, we can run `metron.prefill_profiler`.

Launch any open source system and setup API keys and URL as shown for [vLLM](#running-with-open-source-systems).
```bash
python -m metron.prefill_profiler \
--model "meta-llama/Meta-Llama-3-8B-Instruct" \
--max-num-completed-requests 1 \
--timeout 600 \
--num-ray-clients 1 \
--num-concurrent-requests-per-client 1 \
--fixed-request-generator-decode-tokens 16 \
--output-dir "prefill_experiments/prefill_profiler_vllm_llama-3-8b" \
--should-use-given-dir true
```

To modify range of prompt tokens for which prefill times get profiled, use the flag ``--prefill-lengths`` as follows:
```bash
python -m metron.prefill_profiler \
--model "meta-llama/Meta-Llama-3-8B-Instruct" \
--max-num-completed-requests 1 \
--timeout 600 \
--num-ray-clients 1 \
--num-concurrent-requests-per-client 1 \
--output-dir "prefill_experiments/prefill_profiler_vllm_llama-3-8b" \
--should-use-given-dir true \
--prefill-lengths 256 512 1024 2048 4096 8192 16384 32768 65536
```

## Running Capacity Search
`Important`: Run prefill profiler for a given model and open source system before running capacity search of `deadline-based` SLO type.

Refer to [readme](metron/capacity_search/README.md) file of `metron/capacity_search` folder to know more about how to run capacity search.

## Implementing New LLM Clients

To implement a new LLM client, you need to implement the base class `metron.llm_client.BaseLLMClient` and decorate it as a ray actor.

```python

from metron.llm_client import BaseLLMClient
import ray


@ray.remote
class CustomLLMClient(BaseLLMClient):

    def send_llm_request(self, request_config: RequestConfig) -> Tuple[Metrics, str, RequestConfig]:
        """Make a single completion request to a LLM API

        Returns:
            Metrics about the performance charateristics of the request.
            The text generated by the request to the LLM API.
            The request_config used to make the request. This is mainly for logging purposes.

        """
        ...

```


## Citation
If you use our work, please consider citing our paper:
```cite
@article{agrawal2024metron,
  title={Metron: Holistic Performance Evaluation Framework for LLM Inference Systems},
  author={Agrawal, Amey and Agarwal, Anmol and Kedia, Nitin and Mohan, Jayashree and Kundu, Souvik and Kwatra, Nipun and Ramjee, Ramachandran and Tumanov, Alexey},
  journal={},
  year={2024}
}
```

## Acknowledgement
This repository was originally created as fork from [LLMPerf](https://github.com/ray-project/llmperf) project.

