Interval Generators
===================

Interval generators determine the time interval between consecutive requests. The following interval generators are available in ``etalon``:

Poisson Interval Generator
--------------------------

The Poisson interval generator generates intervals between requests according to a Poisson distribution. To set up the Poisson interval generator, use the following configuration:

.. code-block:: shell

    python -m etalon.run_benchmark
        # other arguments
        ... \
        --request_interval_generator_config_type "poisson" \
        --poisson_request_interval_generator_config_qps 1.0 \
        --seed 42

In the above example, the Poisson interval generator generates intervals between requests according to a Poisson distribution with a mean of 1.0 second. The seed is set to 42 for reproducibility.

Gamma Interval Generator
------------------------

The Gamma interval generator generates intervals between requests according to a Gamma distribution. To set up the Gamma interval generator, use the following configuration:

.. code-block:: shell

    python -m etalon.run_benchmark
        # other arguments
        ... \
        --request_interval_generator_config_type "gamma" \
        --gamma_request_interval_generator_config_cv 1.0 \
        --gamma_request_interval_generator_config_qps 1.0 \
        --seed 42

In the above example, the Gamma interval generator generates intervals between requests according to a Gamma distribution with a coefficient of variation (CV) of 1.0 and a mean of 1.0 second. The seed is set to 42 for reproducibility.

Static Interval Generator
-------------------------

The static interval generator generates no interval between requests, i.e., each request is launched immediately after the previous request. To set up the static interval generator, use the following configuration:

.. code-block:: shell

    python -m etalon.run_benchmark
        # other arguments
        ... \
        --request_interval_generator_config_type "static"

Trace Interval Generator
------------------------

The trace interval generator generates intervals between requests based on a trace file. To set up the trace interval generator, use the following configuration:

.. code-block:: shell

    python -m etalon.run_benchmark
        # other arguments
        ... \
        --request_interval_generator_config_type "trace" \
        --trace_request_interval_generator_config_trace_file "path/to/trace/file" \
        --trace_request_interval_generator_config_start_time "1970-01-04 12:00:00" \
        --trace_request_interval_generator_config_end_time "1970-01-04 15:00:00" \
        --seed 42

In the above example, the trace interval generator generates intervals between requests based on a trace file. The trace file should contain timestamps of requests. The start and end times are used to determine the time range for generating intervals. The seed is set to 42 for reproducibility.
