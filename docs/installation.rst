Installation
============

Setup
-----

Clone Repository
~~~~~~~~~~~~~~~~

.. code-block:: shell

    git clone https://github.com/project-etalon/etalon.git

Create Conda Environment
~~~~~~~~~~~~~~~~~~~~~~~~
.. code-block:: shell

    conda create -n etalon python=3.10
    conda activate etalon

Install etalon
~~~~~~~~~~~~~~~
.. code-block:: shell

    cd etalon
    pip install -e .

Optional Dependencies
^^^^^^^^^^^^^^^^^^^^^
.. code-block:: shell

    pip install -e ".[vllm]"

.. _huggingface_setup:

Setup Hugging Face
~~~~~~~~~~~~~~~~~~

First create and setup your account at ``https://huggingface.co/`` and obtain API key. Then run the following command and enter API key linked to your hugging face account:

.. code-block:: shell

    huggingface-cli login

Custom tokenizer corresponding to the model is fetched from Hugging Face hub. Make sure you have access to the model and are logged in to Hugging Face though command line.

.. _wandb_setup:

Setup Wandb [Optional]
~~~~~~~~~~~~~~~~~~~~~~
First create and setup your account at ``https://<your-org>.wandb.io/`` or public Wandb and obtain API key. Then run the following command and enter API key linked to your wandb account:

.. code-block:: shell

    wandb login --host https://<your-org>.wandb.io

Disabling Wandb
^^^^^^^^^^^^^^^^^^^
To opt out of wandb, do any of the following:

1. Don't pass any wandb related args like ``--wandb-project``, ``--wandb-group`` and ``wandb-run-name`` when running python scripts. Alternatively, pass in ``--no-should-write-metrics`` instead of ``--should-write-metrics`` boolean flag.
2. Run ``export WANDB_MODE=disabled`` in your shell or add this to ``~/.zshrc`` or ``~/.bashrc``. Remember to reload your shell using ``source ~/.zshrc`` or ``source ~/.bashrc``.

