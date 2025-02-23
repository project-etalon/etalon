Proprietary Systems
===================
``etalon`` can benchmark the performance of LLM Inference Systems that are exposed as public APIs. The following sections describe how to benchmark these systems.

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

    python -m etalon.run_benchmark \
    --client_config_model "meta-llama/Meta-Llama-3-8B-Instruct" \
    --max_completed_requests 20 \
    --request_interval_generator_config_type "gamma" \
    --request_length_generator_config_type "zipf" \
    --zipf_request_length_generator_config_max_tokens 8192 \
    --metrics_config_output_dir "results"

Be sure to update ``--client_config_model`` flag to the model used in the proprietary system.

.. note::

    ``etalon`` supports different generator providers for request interval and request length. For more details, refer to :doc:`../guides/request_generator_providers`.

.. _wandb_args_proprietary_systems:

Specifying wandb args [Optional]
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Optionally, you can also specify the following arguments to log results to wandb:

.. code-block:: shell

    --metrics_config_should_write_metrics \
    --metrics_config_wandb_project Project \
    --metrics_config_wandb_group Group \
    --metrics_config_wandb_run_name Run

Other Arguments
^^^^^^^^^^^^^^^
There are many more arguments for running benchmark, run the following to know more:

.. code-block:: shell

    python -m etalon.run_benchmark -h


Saving Results
~~~~~~~~~~~~~~~
The results of the benchmark are saved in the results directory specified by the ``--metrics_config_output_dir`` argument.
