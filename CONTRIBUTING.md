# 🤝 Contributing

Thanks for your interest. Because this repository backs a submitted manuscript, direct pushes to `main` are limited to the author list; everyone else is welcome through pull requests and issues.

## Reporting a bug or asking a question

Open an **issue** at <https://github.com/manuelcortes17/dinov2-phaseolus-boeremia-latent-phenotyping/issues> with:

- the script you ran and the command line used,
- the full traceback or screenshot,
- your OS, Python/R versions, and (if relevant) GPU model.

## Suggesting an improvement

1. Fork the repo and create a branch:
   ```bash
   git checkout -b feature/<short-name>
   ```
2. Follow the existing style:
   - **Python:** PEP-8, matplotlib via `src/figstyle.py`, no inline hex colours.
   - **R:** tidyverse style, snake_case; CSV outputs go in `tables/`.
3. Any new figure must be reproducible from a single script; do not commit screenshots.
4. Submit a pull request to `main` describing *what changed and why*.

## Reproducing the frozen numbers

All reported numbers come from `frozen/frozen_results.json`. Please do **not** overwrite that file unless you are rerunning the whole pipeline — figures and manuscript must agree with it.

## Code of Conduct

Be kind, be curious, give credit. We follow the [Contributor Covenant v2.1](https://www.contributor-covenant.org/version/2/1/code_of_conduct/).
