Solvers
=======

Linear models
-------------

Linear models are useful as they emerge very often in daily life.
When solving fit problem, linear means linear in terms of regression parameters
not features (variables). This distinction is important and explain why we
can linearly regress parabola.

.. automodule:: scifit.solvers.linear
   :members:


Scientific models
-----------------

.. note::

    Your model is missing, make a `PR <https://github.com/jlandercy/scifit/pulls>`_
    we will be glad to add to our collection.

Scientific models is a collection of useful solver for the scientific daily life.
Scientific models are generally non linear in terms of parameters and may require
extra work to solve sub problems before expressing the model function (integration,
solving system of equations, and so on).

.. automodule:: scifit.solvers.scientific
   :members:

