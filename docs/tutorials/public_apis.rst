Proprietary Systems
===================
``metron`` can benchmark the performance of LLM Inference Systems that are exposed as public APIs. The following sections describe how to benchmark these systems.

.. note::

    Custom tokenizer corresponding to the model is fetched from Hugging Face hub. Make sure you have access to the model and are logged in to Hugging Face. Check :ref:`huggingface_setup` for more details.

Export API Key and URL
~~~~~~~~~~~~~~~~~~~~~~
.. code-block:: shell

    export OPENAI_API_BASE=https://api.endpoints.anyscale.com/v1
    export OPENAI_API_KEY=secret_abcdefg

Running Benchmark
~~~~~~~~~~~~~~~~~

.. code-block:: shell

    python -m metron.run_benchmark \
    --model "meta-llama/Meta-Llama-3-8B-Instruct" \
    --max-num-completed-requests 150 \
    --timeout 600 \
    --num-concurrent-requests 10 \
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

There are many more arguments for running benchmark, run the following to know more:

.. code-block:: shell

    python -m metron.run_benchmark -h
