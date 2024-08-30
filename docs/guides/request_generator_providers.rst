Configuring Request Generator Providers
=======================================

``etalon`` supports multiple types of request and interval generators. They are of two types:

1. **Request Interval Generator Provider**: They determine the time interval between consecutive requests.
2. **Request Length Generator Provider**: They determine the number of prompt and decode tokens for each request.

The following sections describe the different request generator providers available in ``etalon``. 

.. toctree::
    :maxdepth: 2

    request_interval_generator_provider
    request_length_generator_provider

