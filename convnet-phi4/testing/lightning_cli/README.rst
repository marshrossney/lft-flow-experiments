=============================
Testing Pytorch Lightning CLI
=============================

This does not seem to be working consistently.

---------------
Version 1
---------------

The script is run as

.. code::
    
    if __name__ == "__main__":
        cli = LightningCLI(parser_kwargs={"error_handler": None})

I do

.. code::

    $ python model.py fit --model Model --print_config >> config_1.yaml
    $ python model.py fit --config config_1.yaml
    ...
    jsonargparse.util.ParserError: 'Configuration check failed :: No action for destination key "fit.model" to check its value.'

So it doesn't like the key ``fit.model``

But then I do it again and for some reason the printed config changes!! See ``config_1.yaml``. Now it seems to work!

Ok...so then I do 

.. code::

    $ python model.py fit --model Model --model.arg 4 --print_config >> config_2.yaml
    $ python model.py fit --config config_2.yaml
    ...
    Configuration check failed :: No action for destination key "arg" to check its value.


But these should be almost exactly the same... (I know ``arg`` should be int ``4`` not string ``'4'`` but this isn't the problem)

.. code::

    $ diff config_1.yaml config_2.yaml
    <   init_args: {}
    ---
    >   init_args:
    >     arg: '4'



---------------
Version 2
---------------

The script is run as

.. code::
    
    if __name__ == "__main__":
        cli = LightningCLI(Model, parser_kwargs={"error_handler": None})

Now just a simple ``python model.py fit`` works since the model is specified inside the script itself.

However, ``python model fit --model.arg 10`` does not work, nor does ``--fit.model/arg`` or just ``--arg``, so I have no idea how to provide cmd args. I also can't add anything to the config file ``config_3.yaml`` without it breaking.
