Interfaces
==========

.. automodule:: newproject.interfaces

Generic Interface
-----------------

.. automodule:: newproject.interfaces.generic

.. autoclass:: newproject.interfaces.generic.GenericInterface
   :members:

Implementation examples
-----------------------

.. automodule:: newproject.interfaces.examples

Simple Cases
************

In this use case, two basic interfaces are defined. The class :class:`SimpleCase`
which accepts a single value as configuration but without any serializer defined:

.. autoclass:: newproject.interfaces.examples.SimpleCase
   :members:

And the class :class:`SimpleCaseWithSerializer` which inherits from :class:`SimpleCase`
and do provide a serializer:

.. autoclass:: newproject.interfaces.examples.SimpleCaseWithSerializer
   :members:
