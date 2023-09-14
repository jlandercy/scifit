Installation
############

SciFit Package is available on `PyPi repository <https://pypi.org/manage/project/scifit/releases/>`_:

.. code-block:: bash

   python -m pip --upgrade install scifit

Or it can be installed using the usual package setup flow:

.. code-block:: bash

   python setup.py install

After downloading source code from `GitHub repository <https://github.com/jlandercy/scifit>`_.

Suites
******

Additional capabilities are exposed, first install tool suite:

.. code-block:: bash

   python -m pip install -r requirements_ci.txt

Then you can run all suites.

To package and install from a wheel:

.. code-block:: bash

   nox -s package install

To uninstall the package:

.. code-block:: bash

   nox -s uninstall

To check unit tests suite:

.. code-block:: bash

   nox -s tests

To build documentation:

.. code-block:: bash

   nox -s notebooks docs
