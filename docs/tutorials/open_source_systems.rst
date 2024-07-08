Open Source Systems
===================

``metron`` can be run with any open source LLM inference system. If open source system does not provide OpenAI Compatible APIs, then new LLM clients can be implemented to support new open source system as explained in :doc:`../guides/new_llm_client`.

.. note::

    Custom tokenizer corresponding to the model is fetched from Hugging Face hub. Make sure you have access to the model and are logged in to Hugging Face. Check :ref:`huggingface_setup` for more details.

Here we give an example with ``vLLM``.

Launch vLLM Server
~~~~~~~~~~~~~~~~~~

.. code-block:: shell

    python -m vllm.entrypoints.openai.api_server --model meta-llama/Meta-Llama-3-8B-Instruct --dtype auto --api-key token-abc123 -tp 1 --rope-scaling '{"type":"dynamic","factor":2.0}'

If higher context length is needed than supported by the model with certain scale factor, then add rope-scaling as ``--rope-scaling '{"type":"dynamic","factor":2.0}'``. Adjust type and factor as per the use case.

Export API Key and URL
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: shell

    export OPENAI_API_BASE=http://localhost:8000/v1
    export OPENAI_API_KEY=token-abc123

Running Benchmark
~~~~~~~~~~~~~~~~~
Benchmark can be run as shown below:

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

Be sure to update ``--model`` flag to same model used to launch vLLM.

Saving Results
~~~~~~~~~~~~~~~
The results of the benchmark are saved in the results directory specified by the ``--output-dir`` argument.
