
Installation guide
==================

Prerequisites
-------------


* OS support: Windows, Linux and OSX

Dependencies
------------

.. list-table::
   :header-rows: 1

   * - name
     - version
   * - numpy
     - -
   * - pandas
     - -
   * - paddlepaddle
     - &gt;=2.0.0rc0
   * - pgl
     - &gt;=2.1


('-' means no specific version requirement for that package)

Instruction
-----------

PaddleSpatial depends on the ``paddlepaddle`` of version 2.0.0rc0 or above. We suggest using ``conda`` to create a new environment for the installation. Detailed instruction is shown below:


#. 
   If you do not have conda installed, please check this website to get it first:

   https://docs.conda.io/projects/conda/en/latest/user-guide/install/

#. 
   Create a new environment with conda:

   .. code-block:: bash

      conda create -n paddlespatial


#. 
   Activate the environment which is just created:

   .. code-block:: bash

      conda activate paddlespatial


#. 
   Install the right version of ``paddlepaddle`` according to the device (CPU/GPU) you want to run PaddleSpatial on.

   If you want to use the GPU version of ``paddlepaddle``\ , run this:

   .. code-block:: bash

      python -m pip install paddlepaddle-gpu -f https://paddlepaddle.org.cn/whl/stable.html

    Or if you want to use the CPU version of ``paddlepaddle``\ , run this:

   .. code-block:: bash

      python -m pip install paddlepaddle -i https://mirror.baidu.com/pypi/simple

   Noting that the version of ``paddlepaddle`` should be higher than **2.0**.
   Check ``paddlepaddle``\ 's `official document <https://www.paddlepaddle.org.cn/documentation/docs/en/2.0-rc1/install/index_en.html>`_
   for more installation guide.

#. 
   Install ``PGL`` using pip:

   .. code-block:: bash

      pip install pgl


#. 
   Install PaddleSpatial using pip:

   .. code-block:: bash

      pip install paddlespatial


#. 
   The installation is done!

Note
^^^^

After playing, if you want to deactivate the conda environment, do this:

   .. code-block:: bash

      conda deactivate

