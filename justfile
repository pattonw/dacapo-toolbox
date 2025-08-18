docs:
    uv run --extra docs --pre --python 3.11 python -m ipykernel install --name dacapo_env --user
    uv run --extra docs --pre --python 3.11 jupytext --to notebook tutorials/dataset/dataset_overview.py
    uv run --extra docs --pre --python 3.11 jupytext --to notebook tutorials/dataset/dataset_training.py
    cp -r tutorials docs/source
    uv run --extra docs --pre --python 3.11 sphinx-build docs/source docs/build/html -b html

docs-clean:
    rm -rf docs/build/html
    rm -rf docs/source/tutorials