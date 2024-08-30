.. etalon documentation master file, created by
   sphinx-quickstart on Sat Jul  6 17:47:44 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

etalon
=======

   A tool for benchmarking the performance of LLM Inference Systems.

Serving large language models (LLMs) in production is very expensive, and this challenge has prompted recent advances in inference system optimizations. As of today, these systems are evaluated through conventional metrics like TTFT (time to first token), TBT (time between tokens), normalized latency, and TPOT (time per output token). However, these metrics fail to capture the nuances of LLM inference leading to incomplete assessment of user-facing performance in real-time applications like chat.

``etalon`` is a holistic performance evaluation framework that includes new metrics, :ref:`fluidity-index` and :ref:`fluid-token-generation-rate`, alongside existing conventional metrics. The new metrics reflect the intricacies of LLM inference process and its impact on real-time user experience.

``etalon`` is designed to be easy to use, and can be integrated with any LLM inference system. It is built on top of Ray, a popular distributed computing framework, and can be used to benchmark LLM inference systems on a single machine or a cluster.

Check out the following resources to get started:

.. toctree::
   :maxdepth: 2

   installation
   tutorials/metrics
   how_to_use
   guides/guide

Citation
--------

If you use our work, please consider citing our paper `etalon: Holistic Performance Evaluation Framework for LLM Inference Systems <https://arxiv.org/abs/2407.07000>`_:

.. code-block:: bibtex

      @misc{agrawal2024etalonholisticperformanceevaluation,
         title={etalon: Holistic Performance Evaluation Framework for LLM Inference Systems}, 
         author={Amey Agrawal and Anmol Agarwal and Nitin Kedia and Jayashree Mohan and Souvik Kundu and Nipun Kwatra and Ramachandran Ramjee and Alexey Tumanov},
         year={2024},
         eprint={2407.07000},
         archivePrefix={arXiv},
         primaryClass={cs.LG},
         url={https://arxiv.org/abs/2407.07000}, 
      }

Acknowledgement
---------------
`etalon <https://github.com/project-etalon/etalon>`_ code was originally created as a fork from `LLMPerf <https://github.com/ray-project/llmperf>`_ project.

