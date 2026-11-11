window.BENCHMARK_DATA = {
  "lastUpdate": 1794425843295,
  "repoUrl": "https://github.com/pattonw/dacapo-toolbox",
  "entries": {
    "dacapo-toolbox": [
      {
        "commit": {
          "author": {
            "name": "William Patton",
            "username": "pattonw",
            "email": "wllmpttn24@gmail.com"
          },
          "committer": {
            "name": "GitHub",
            "username": "web-flow",
            "email": "noreply@github.com"
          },
          "id": "4c9ac9062548bd1a3b04b3d77e141c1c21eaf522",
          "message": "Run coverage from pytest addopts; make the postprocessing module importable (#11)\n\n* Run coverage from pytest addopts so a bare `uv run pytest` is the CI command\n\n`[tool.pytest.ini_options] addopts` now carries `--cov --cov-report=term-missing`,\nso the README's `uv run pytest` prints the coverage table and enforces\n`[tool.coverage.report] fail_under` the same way CI does. The CI step drops the\nflags it no longer needs. A single-file run can skip the floor with `--no-cov`.\n\nCo-Authored-By: Claude Mythos 5.1 <noreply@anthropic.com>\n\n* Make the postprocessing module importable: volara-ml, Python 3.12, recent floors\n\n`import dacapo_toolbox.postprocessing` failed in the locked Python 3.11\nenvironment: gunpowder 1.5 needs Python 3.12, so the 3.11 side of the lock\nkept gunpowder 1.4.3 with zarr 2 and therefore volara 1.0.2, which imports\nfunlib.math without declaring it. Nothing in CI imported the module, so main\nstayed green with a public module that did not import on the floor.\n\nThe module now depends on volara-ml (the renamed volara-torch) and the package\non the zarr 3 generation of its stack: Python 3.12, gunpowder 1.5, funlib\npersistence 0.8.1, volara 2.0.1, volara-ml 1.2 with the torch extra. The lock\nno longer forks on the interpreter version. `.python-version` pins the\nproject environment to the floor so that `uv sync --all-extras` installs the\nlocked numba, which has no 3.14 wheel yet; CI installs the postprocessing\nextra on the 3.12 jobs for the same reason.\n\ntests/test_postprocessing.py runs `blockwise_predict_mutex` end to end on two\nbright cubes over a dark background with a hand-built affinity model, and\nasserts that each cube becomes one segment, the segments differ, and the\nbackground two voxels away from either cube stays unlabelled. It skips with\na reason naming volara when the extra is not installed.\n\nCo-Authored-By: Claude Mythos 5.1 <noreply@anthropic.com>\n\n* Track import time and test duration across commits\n\nThe tests job times its own `uv run pytest`, measures the cold import of\nthe package's headline modules (benchmarks/run.py), and uploads the two\nnumbers as an artifact from one matrix cell. The benchmark workflow runs\nwhen a tests run on main completes, downloads that artifact, creates the\ngh-pages branch if it is missing, and publishes with github-action-benchmark\nto gh-pages:dev/bench, with a comparison table in the job summary and a\ncommit comment when a metric gets 1.5x worse. The docs gain a Benchmarks\npage that charts every metric from that data, one line per metric with the\ncommit on hover and click; the docs workflow now runs after each benchmark\nrun and copies dev/bench from gh-pages into the deployed site, so the\ncharts carry the newest point and the action's own dashboard keeps its URL.\nNothing runs the test suite twice.\n\nCo-Authored-By: Claude Mythos 5.1 <noreply@anthropic.com>\n\n---------\n\nCo-authored-by: Claude Mythos 5.1 <noreply@anthropic.com>",
          "timestamp": "2026-11-05T03:36:19Z",
          "url": "https://github.com/pattonw/dacapo-toolbox/commit/4c9ac9062548bd1a3b04b3d77e141c1c21eaf522"
        },
        "date": 1793849923346,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "import dacapo_toolbox.dataset (cold)",
            "value": 3138.999,
            "range": "± 12.867",
            "unit": "ms",
            "extra": "interpreter startup 12.0 ms subtracted; n=10"
          },
          {
            "name": "pytest wall time",
            "value": 53.625,
            "unit": "s",
            "extra": "`uv run pytest --cov-report=xml` in the tests job"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "William Patton",
            "username": "pattonw",
            "email": "wllmpttn24@gmail.com"
          },
          "committer": {
            "name": "GitHub",
            "username": "web-flow",
            "email": "noreply@github.com"
          },
          "id": "c3073c36dada8e1c5f4a0ead6df9430ba9941673",
          "message": "Deploy docs on version tags and by hand only (#12)\n\n* Deploy docs on version tags and by hand only\n\nThe docs workflow now runs on a v* tag push and on workflow_dispatch, and\nnot on every push or pull request commit. A tag or a manual run on main\ndeploys; a manual run on any other branch builds without deploying, which\nis how a pull request's docs are checked once it is ready to merge.\n\nCo-Authored-By: Claude Fable 5.2 <noreply@anthropic.com>\n\n* Draw the benchmark charts on pages that load require.js\n\nThe notebook extension puts require.js on the docs pages. With an AMD\nloader present, the Chart.js bundle registers as an AMD module and never\nsets the global Chart, so the benchmarks page stayed empty although the\ndata was deployed. Hide define while Chart.js loads and restore it after.\n\nCo-Authored-By: Claude Fable 5.2 <noreply@anthropic.com>\n\n* Run the lowest-direct test leg against the declared floors\n\nThe resolution mode was passed to `uv sync` only. The next step, a plain\n`uv run pytest`, saw a lockfile written in another mode, re-resolved to the\nnewest versions and replaced the environment, so the leg named\nlowest-direct tested unlocked newest versions instead. The mode is now set\nonce as UV_RESOLUTION for the whole job.\n\nCo-Authored-By: Claude Fable 5.2 <noreply@anthropic.com>\n\n* Depend on volara 2.1.1\n\nvolara 2.1.1 excludes cloud-volume 12.16.0, which fails at import without\ncrackle-codec, and loads cloud-volume only on first use. With an older\nvolara floor, the lowest-direct test leg installs that volara together\nwith the newest cloud-volume and fails at collection.\n\nCo-Authored-By: Claude Fable 5.2 <noreply@anthropic.com>\n\n---------\n\nCo-authored-by: Claude Fable 5.2 <noreply@anthropic.com>",
          "timestamp": "2026-11-11T19:30:49Z",
          "url": "https://github.com/pattonw/dacapo-toolbox/commit/c3073c36dada8e1c5f4a0ead6df9430ba9941673"
        },
        "date": 1794425577234,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "import dacapo_toolbox.dataset (cold)",
            "value": 2653.83,
            "range": "± 42.269",
            "unit": "ms",
            "extra": "interpreter startup 9.0 ms subtracted; n=10"
          },
          {
            "name": "pytest wall time",
            "value": 37.717,
            "unit": "s",
            "extra": "`uv run pytest --cov-report=xml` in the tests job"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "Will",
            "username": "pattonw",
            "email": "wllmpttn24@gmail.com"
          },
          "committer": {
            "name": "Will",
            "username": "pattonw",
            "email": "wllmpttn24@gmail.com"
          },
          "id": "c7c468f8b764daeb35dccf700ca1918209a7929f",
          "message": "Bump version to 1.1.0\n\nCo-Authored-By: Claude Fable 5.2 <noreply@anthropic.com>",
          "timestamp": "2026-11-11T19:34:54Z",
          "url": "https://github.com/pattonw/dacapo-toolbox/commit/c7c468f8b764daeb35dccf700ca1918209a7929f"
        },
        "date": 1794425831296,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "import dacapo_toolbox.dataset (cold)",
            "value": 3111.207,
            "range": "± 38.407",
            "unit": "ms",
            "extra": "interpreter startup 10.9 ms subtracted; n=10"
          },
          {
            "name": "pytest wall time",
            "value": 50.714,
            "unit": "s",
            "extra": "`uv run pytest --cov-report=xml` in the tests job"
          }
        ]
      }
    ]
  }
}