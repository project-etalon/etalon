Capacity Evaluation
===================

.. toctree::
    :maxdepth: 2
    :hidden:

    prefill_profiler
    capacity_search

``metron`` provides a tool through capacity search to find the maximum Queries per Second (QPS) that a given model and inference system can handle given various constraints.

What is Capacity?
-----------------

It is defined as maximum request load (queries-per-second) a system can sustain while meeting certain latency targets (SLOs). Higher capacity reduces the cost of serving requests and improves the user experience.

Steps to Measure Capacity
-------------------------

Prefill Profiler
~~~~~~~~~~~~~~~~
First ``metron`` needs to profile the prefill times of the given open source system and model combination. 

Refer to :doc:`prefill_profiler` for more details on how to run prefill profiler.


Capacity Search
~~~~~~~~~~~~~~~
``metron`` then runs capacity search to find the maximum QPS. Capacity search also helps in generation of :ref:`fluid-token-generation-rate` metric.

Refer to :doc:`capacity_search` for more details on how to run capacity search.
