.. metron documentation master file, created by
   sphinx-quickstart on Sat Jul  6 17:47:44 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

metron
=======

   A tool for benchmarking the performance of LLM Inference Systems.

Serving large language models (LLMs) in production is very expensive, and this challenge has prompted recent advances in inference system optimizations. As of today, these systems are evaluated through conventional metrics like TTFT (time to first token), TBT (time between tokens), normalized latency, and TPOT (time per output token). However, these metrics fail to capture the nuances of LLM inference leading to incomplete assessment of user-facing performance in real-time applications like chat.

``metron`` is a holistic performance evaluation framework that includes new metrics, :ref:`fluidity-index` and :ref:`fluid-token-generation-rate`, alongside existing conventional metrics. The new metrics reflect the intricacies of LLM inference process and its impact on real-time user experience.

``metron`` is designed to be easy to use, and can be integrated with any LLM inference system. It is built on top of Ray, a popular distributed computing framework, and can be used to benchmark LLM inference systems on a single machine or a cluster.

Check out the following resources to get started:

.. toctree::
   :maxdepth: 2

   installation
   tutorials/metrics
   how_to_use
   guides/guide

Citation
--------

If you use our work, please consider citing our paper:

.. code-block:: bibtex

      @article{agrawal2024metron,
         title={Metron: Holistic Performance Evaluation Framework for LLM Inference Systems},
         author={Agrawal, Amey and Agarwal, Anmol and Kedia, Nitin and Mohan, Jayashree and Kundu, Souvik and Kwatra, Nipun and Ramjee, Ramachandran and Tumanov, Alexey},
         journal={},
         year={2024}
      }

Acknowledgement
---------------
`metron <https://github.com/gatech-sysml/metron>`_ code was originally created as a fork from `LLMPerf <https://github.com/ray-project/llmperf>`_ project.

