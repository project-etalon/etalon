Length Generators
=================

Length generators determine the number of prompt and decode tokens for each request. The following length generators are available in ``etalon``:

Uniform Length Generator
------------------------

The uniform length generator generates the number of prompt and decode tokens according to a uniform distribution. To set up the uniform length generator, use the following configuration:

.. code-block:: shell

    python -m etalon.run_benchmark
        # other arguments
        ... \
        --request_length_generator_config_type "uniform" \
        --uniform_request_length_generator_config_min_tokens 128 \
        --uniform_request_length_generator_config_max_tokens 256 \
        --uniform_request_length_generator_config_prefill_to_decode_ratio 0.5 \
        --seed 42
        
In the above example, the uniform length generator generates the total number of tokens according to a uniform distribution with a minimum of 128 tokens and a maximum of 256 tokens. The prefill-to-decode ratio is set to 0.5. Which means 50% of total tokens would be prefill tokens and rest would be decode tokens. The seed is set to 42 for reproducibility.

Zipf Length Generator
---------------------

The Zipf length generator generates the number of prompt and decode tokens according to a Zipf distribution. To set up the Zipf length generator, use the following configuration:

.. code-block:: shell

    python -m etalon.run_benchmark
        # other arguments
        ... \
        --request_length_generator_config_type "zipf" \
        --zipf_request_length_generator_config_theta 0.4 \
        [--no-zipf_request_length_generator_config_scramble | --zipf_request_length_generator_config_scramble] \
        --zipf_request_length_generator_config_min_tokens 128 \
        --zipf_request_length_generator_config_max_tokens 256 \
        --zipf_request_length_generator_config_prefill_to_decode_ratio 0.5 \
        --seed 42

In the above example, the Zipf length generator generates the total number of tokens according to a Zipf distribution with a theta of 0.4. The scramble flag is used to scramble the Zipf distribution. The minimum number of tokens is set to 128, and the maximum number of tokens is set to 256. The prefill-to-decode ratio is set to 0.5. The seed is set to 42 for reproducibility.

Trace Length Generator
----------------------

The trace length generator generates the number of prompt and decode tokens according to a trace. To set up the trace length generator, use the following configuration:

.. code-block:: shell

    python -m etalon.run_benchmark
        # other arguments
        ... \
        --request_length_generator_config_type "trace" \
        --trace_request_length_generator_config_trace_file "path/to/trace/file" \
        --trace_request_length_generator_config_prefill_scale_factor 0.5 \
        --trace_request_length_generator_config_decode_scale_factor 0.5 \
        --trace_request_length_generator_config_max_tokens 512 \
        --seed 42

In the above example, the trace length generator generates the total number of tokens according to a trace file. The prefill scale factor is set to 0.5, and the decode scale factor is set to 0.5. The maximum number of tokens is set to 512. The seed is set to 42 for reproducibility.

Fixed Length Generator
----------------------

The fixed length generator generates the number of prompt and decode tokens according to fixed values given as input. To set up the fixed length generator, use the following configuration:

.. code-block:: shell

    python -m etalon.run_benchmark
        # other arguments
        ... \
        --request_length_generator_config_type "fixed" \
        --fixed_request_length_generator_config_prefill_tokens 128 \
        --fixed_request_length_generator_config_decode_tokens 128 \
        --seed 42

In the above example, the fixed length generator generates the total number of tokens according to fixed values. The prefill tokens are set to 128, and the decode tokens are set to 128. The seed is set to 42 for reproducibility.

