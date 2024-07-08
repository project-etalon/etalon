Getting Started
===============

.. toctree::
   :maxdepth: 2
   :hidden:

   tutorials/blackbox_evaluation
   tutorials/measuring_qps
   tutorials/metrics

``metron`` can evaluate LLM inference systems as a black-box and also determine the serving capacity of the LLM inference system.

``metron`` provides two evaluation recipes as described below:

- **Black-box Evaluation**: ``metron`` hits LLM inference server exposed through API endpoint with set of requests with different prompt lengths and tracks when each output token is generated. This allows ``metron`` to calculate several metrics like TTFT, TBT, TPOT, and *unified-deadline*. With *unified-deadline* metric, ``metron`` also infers the minimum TBT required to meet target acceptance rate threshold, such as - 99% of requests should have acceptance rate of at least 90%. 

- **Capacity Evaluation**: When deploying LLM inference system, operator needs to know how many requests can be served by the system. This will help operator in determining the configuration of the system, for example, the number of GPUs needed, to meet certain service quality requirements. To help with this process, ``metron`` provides a capacity evaluation module which determines maximum capacity each replica can provide under different request loads while meeting target SLO requirements.


The description of each metric used in black-box evaluation is provided in :doc:`tutorials/metrics`.

Check out the following resources to learn more:

* :doc:`tutorials/blackbox_evaluation`
* :doc:`tutorials/measuring_qps`
* :doc:`tutorials/metrics`
