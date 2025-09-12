docs:
    uv run --all-extras --python 3.11 python -m ipykernel install --name dacapo_env --user
    uv run --all-extras --python 3.11 jupytext --to notebook tutorials/dataset/dataset_overview.py
    uv run --all-extras --python 3.11 jupytext --to notebook tutorials/dataset/cremi.py
    cp -r tutorials docs/source
    rm docs/source/tutorials/*/*.py
    uv run --all-extras --python 3.11 sphinx-build docs/source docs/build/html -b html

docs-clean:
    rm -rf docs/build/html
    rm -rf docs/source/tutorials