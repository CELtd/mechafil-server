# MechaFil Server Documentation

This directory contains the complete documentation for the MechaFil Server REST API.

## Building the Documentation

### Prerequisites

Install the documentation dependencies:

```bash
poetry install --with docs
```

### Build HTML Documentation

```bash
cd docs
make html
```

The built documentation will be in `docs/build/html/`. Open `docs/build/html/index.html` in your browser.

### Build PDF Documentation

```bash
cd docs
make latexpdf
```

The PDF will be generated in `docs/build/latex/`.

### Live Auto-Reload (Development)

For automatic rebuilding while editing:

```bash
poetry run sphinx-autobuild docs/source docs/build/html
```

Then visit http://127.0.0.1:8000

### Clean Build Files

```bash
cd docs
make clean
```

## Documentation Structure

```
docs/
├── source/
│   ├── index.rst              # Main documentation index
│   ├── conf.py                # Sphinx configuration
│   ├── api/
│   │   ├── endpoints.rst      # API endpoint reference
│   │   └── models.rst         # Data models documentation
│   ├── examples/
│   │   ├── quickstart.rst     # Quick start guide
│   │   └── advanced.rst       # Advanced usage patterns
│   ├── configuration.rst      # Server configuration guide
│   ├── deployment.rst         # Deployment guide
│   ├── _static/               # Static files (images, CSS)
│   └── _templates/            # Custom templates
├── Makefile                   # Build commands
└── README.md                  # This file
```

## Documentation Sections

- **API Reference** (`api/`)
  - Complete endpoint documentation
  - Request/response models
  - Parameter descriptions
  - Error codes

- **Examples** (`examples/`)
  - Quick start guide with basic examples
  - Advanced usage patterns
  - Integration examples (Python, JavaScript)
  - Visualization examples

- **Configuration** (`configuration.rst`)
  - Environment variables
  - Server settings
  - Cache configuration
  - Production setup

- **Deployment** (`deployment.rst`)
  - Docker deployment
  - Kubernetes setup
  - Cloud providers (AWS, GCP, Azure)
  - Reverse proxy configuration
  - Monitoring and observability

## Read the Docs

This documentation is configured for Read the Docs hosting.

### Setup on Read the Docs

1. Import the repository on Read the Docs
2. The `.readthedocs.yaml` file will configure the build
3. Documentation will be automatically built on each commit

### Configuration

The Read the Docs configuration is in `.readthedocs.yaml` at the repository root.

## Contributing to Documentation

### Adding New Pages

1. Create a new `.rst` file in the appropriate directory
2. Add the file to the table of contents in `index.rst`:

```rst
.. toctree::
   :maxdepth: 2

   path/to/newpage
```

### Formatting Guidelines

- Use **reStructuredText** (.rst) format
- Follow existing page structure
- Include code examples with proper syntax highlighting
- Use cross-references for links between pages
- Add appropriate section headers (=, -, ~, ^)

### Code Blocks

Python example:

```rst
.. code-block:: python

   import requests
   response = requests.get("http://localhost:8000/health")
```

Bash example:

```rst
.. code-block:: bash

   curl http://localhost:8000/health
```

### Cross-References

Link to other documentation pages:

```rst
See :doc:`../api/endpoints` for more details.
```

## Useful Commands

```bash
# Install docs dependencies
poetry install --with docs

# Build HTML
cd docs && make html

# Build PDF
cd docs && make latexpdf

# Auto-rebuild on changes
poetry run sphinx-autobuild docs/source docs/build/html

# Clean build files
cd docs && make clean

# Check for broken links
cd docs && make linkcheck

# View available make targets
cd docs && make help
```

## Resources

- [Sphinx Documentation](https://www.sphinx-doc.org/)
- [reStructuredText Primer](https://www.sphinx-doc.org/en/master/usage/restructuredtext/basics.html)
- [Read the Docs](https://docs.readthedocs.io/)
- [Sphinx RTD Theme](https://sphinx-rtd-theme.readthedocs.io/)
