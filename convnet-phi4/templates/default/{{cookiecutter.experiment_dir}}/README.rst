============================================
{{ cookiecutter._experiment.title | title }}
============================================

Description
{{ cookiecutter._experiment.description | indent(2, True) }}

-------------------
Run this experiment
-------------------

{% if cookiecutter._repo.name | length -%}
(1) Clone this repository, checkout this branch, and navigate to this directory.
(2) Run ``git checkout {{ cookiecutter._repo.commit }}``.
(3) Finally, execute the commands below
{%- endif %}

.. code:: sh

{% for cmd in cookiecutter._experiment.commands -%}
{{ cmd | indent(4, True) }}
{% endfor %}

***

This experiment was created on {{ cookiecutter._date }} by {{ cookiecutter._author.name }} using `PyRex <https://github.com/marshrossney/pyrex>`.
