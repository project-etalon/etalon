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
    --model "meta-llama/Meta-Llama-3-8B-Instruct" \
    --max-num-completed-requests 20 \
    --request-interval-generator-provider "gamma" \
    --request-length-generator-provider "zipf" \
    --request-generator-max-tokens 8192 \
    --output-dir "results"

Be sure to update ``--model`` flag to the model used in the proprietary system.

.. note::

    ``etalon`` supports different generator providers for request interval and request length. For more details, refer to :doc:`../guides/request_generator_providers`.

.. _wandb_args_proprietary_systems:

Specifying wandb args [Optional]
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Optionally, you can also specify the following arguments to log results to wandb:

.. code-block:: shell

    --should-write-metrics \
    --wandb-project Project \
    --wandb-group Group \
    --wandb-run-name Run

Other Arguments
^^^^^^^^^^^^^^^

There are many more arguments for running benchmark, run the following to know more:

.. code-block:: shell

    python -m etalon.run_benchmark -h
