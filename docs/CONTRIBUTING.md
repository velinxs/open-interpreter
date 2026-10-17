# ●

**Open Interpreter is large, open-source initiative to build a standard interface between language models and computers.**

There are many ways to contribute, from helping others on [Github](https://github.com/endolith/open-interpreter/issues) or [Discord](https://discord.gg/6p3fD6rBVm), writing documentation, or improving code.

We depend on contributors like you. Let's build this.

## What should I work on?

First, please familiarize yourself with our [project scope](ROADMAP.md#whats-in-our-scope). Then, pick up a task from our [roadmap](ROADMAP.md) or work on solving an [issue](https://github.com/endolith/open-interpreter/issues).

If you encounter a bug or have a feature in mind, don't hesitate to [open a new issue](https://github.com/endolith/open-interpreter/issues/new/choose).

## Philosophy

This is a minimalist, **tightly scoped** project that places a premium on simplicity. We're skeptical of new extensions, integrations, and extra features. We would rather not extend the system if it adds nonessential complexity.

## Contribution Guidelines

1. Before taking on significant code changes, please discuss your ideas on [Discord](https://discord.gg/6p3fD6rBVm) to ensure they align with our vision. We want to keep the codebase simple and unintimidating for new users.
2. Fork the repository and create a new branch for your work.
3. Follow the [Running Your Local Fork](CONTRIBUTING.md#running-your-local-fork) guide below.
4. Make changes with clear code comments explaining your approach. Try to follow existing conventions in the code.
5. Follow the [Code Formatting and Linting](CONTRIBUTING.md#code-formatting-and-linting) guide below.
6. Open a PR into `main` linking any related issues. Provide detailed context on your changes.

We will review PRs when possible and work with you to integrate your contribution. Please be patient as reviews take time. Once approved, your code will be merged.

## Running Your Local Fork

**Note: the optional extras are not installed by default, so `--os` mode throws import errors without them. Add the extra you need: `uv sync --extra os`, `uv sync --extra local`, or both. You can edit the system messages for these modes in `interpreter/terminal_interface/profiles/defaults`.**

Once you've forked the code and created a new branch for your work, you can run the fork in CLI mode by following these steps:

1. CD into the project folder by running `cd open-interpreter`.
2. Install `uv` [according to their documentation](https://docs.astral.sh/uv/getting-started/installation/). It creates the virtual environment and resolves dependencies.
3. Install dependencies by running `uv sync --group dev --extra server`.
4. Run the program with `uv run interpreter`. Run the tests with `uv run pytest -m "not integration" -q`.

The integration marker is excluded on purpose: those tests call a real model and bill a real API key.

### Installing New Dependencies

To add a runtime dependency, use `uv add package-name`.

### Installing Developer Dependencies

For dependencies only used while developing — testing and formatting tools — use `uv add --group dev package-name`.

## Code Formatting and Linting

This project uses [`ruff`](https://docs.astral.sh/ruff/) for both formatting and linting, configured in `pyproject.toml` (line length 120). The same two commands are what CI runs, so running them before you push is the whole check:

```bash
uv run ruff check interpreter tests scripts
uv run ruff format interpreter tests scripts
```

A [`pre-commit`](https://pre-commit.com/) hook will run these for you on every commit:

```bash
uv run pre-commit install
```

If you need to bypass it for some reason, you can `git commit` with the `--no-verify` flag.

## Licensing

Contributions to Open Interpreter would be under the MIT license before version 0.2.0, or under AGPL for subsequent contributions.

## Questions?

Join our [Discord community](https://discord.gg/6p3fD6rBVm) and post in the #General channel to connect with contributors. We're happy to guide you through your first open source contribution to this project!

**Thank you for your dedication and understanding as we continue refining our processes. As we explore this extraordinary new technology, we sincerely appreciate your involvement.**
