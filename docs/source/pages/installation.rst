Installation
############

SciFit package can be installed using the usual setup flow:

.. code-block:: bash

   python setup.py install

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
