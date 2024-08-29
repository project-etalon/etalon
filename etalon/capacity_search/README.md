# Capacity Search

Tool to help find maximal QPS given different SLOs. There are three types of SLOs:
1. Deadline based: does QPS search based on deadline slo and deadline miss rate slo. Also leverages deadline-miss-rate percentile.
2. TBT-TTFT based: does QPS search based on tbt and ttft slo with their percentiles.
3. TTFT-TPOT based: does QPS search based on ttft and tpot slo with their percentiles.

`Important`: Please run prefill profiler before running capacity search for a given model and open source system as explained [here](../../../README.md#run-prefill-profiler).

## Deadline based Capacity Search
```bash
python -m etalon.capacity_search.main \
--output-dir "cap_experiments/capacity_search/" \
--profile-dir "prefill_experiments/prefill_profiler_vllm_llama-3-8b" \
--slo-type deadline \
--tbt-slo 0.03 \
--ttft-slack-slo 0.3 \
--deadline-miss-rate-slo 0.1 \
--deadline-miss-rate-percentile 0.99 \
--max-iterations 10 \
--config-path ./etalon/capacity_search/config/llama_8b.yml
```
`--profile-dir` should point to where `prefill_predictor.pkl` model is stored for a given model and open source system.

## TBT-TTFT based Capacity Search
```bash
python -m etalon.capacity_search.main \
--output-dir "cap_experiments/capacity_search/" \
--slo-type tbt_ttft \
--tbt-slo 0.03 \
--tbt-percentile 0.9 \
--ttft-slo 0.3 \
--ttft-percentile 0.9 \
--max-iterations 10 \
--config-path ./etalon/capacity_search/config/llama_8b.yml
```

## TTFT-TPOT based Capacity Search
```bash
python -m etalon.capacity_search.main \
--output-dir "cap_experiments/capacity_search/" \
--slo-type ttft_tpot \
--ttft-slo 0.3 \
--ttft-percentile 0.9 \
--tpot-slo 0.03 \
--tpot-percentile 0.9 \
--max-iterations 10 \
--config-path ./etalon/capacity_search/config/llama_8b.yml
```

## Caching
The capacity search runs for given model and open source system are cached. This means, when we run capacity search again with different SLO type and values, the benchmark runs with previously explored QPS values will be used directly instead of doing new benchmark runs.
