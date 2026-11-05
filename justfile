docs:
    uv run --all-extras --group docs python -m ipykernel install --name dacapo_env --user
    uv run --all-extras --group docs jupytext --to notebook tutorials/dataset/dataset_overview.py
    uv run --all-extras --group docs jupytext --to notebook tutorials/dataset/cremi.py
    cp -r tutorials docs/source
    rm docs/source/tutorials/*/*.py
    uv run --all-extras --group docs sphinx-build docs/source docs/build/html -b html

docs-clean:
    rm -rf docs/build/html
    rm -rf docs/source/tutorials