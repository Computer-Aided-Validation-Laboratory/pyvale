# Contributing to Pyvale

Thank you for wanting to contribute to the development of pyvale. Before contributing, please read this page, along with our [Code of Conduct](https://github.com/Computer-Aided-Validation-Laboratory/pyvale/blob/main/CODE_OF_CONDUCT.md) and [Security](https://github.com/Computer-Aided-Validation-Laboratory/pyvale/blob/main/SECURITY.md) guidelines.


## Reporting Security Concerns

If you see something that you believe may be a security concern in Pyvale, please follow the guidance on our [security page](https://github.com/Computer-Aided-Validation-Laboratory/pyvale/blob/main/SECURITY.md). Do **not** open a public GitHub issue for security concerns. 

## Reporting Bugs

If you find a bug that is not a security concern, report it on our [issues page](https://github.com/Computer-Aided-Validation-Laboratory/pyvale/issues). Use the "bug" tag to use the associated bug template to ensure the bug is captured in detail.

## Making Feature Requests

We encourage you to share your ideas for adding features to Pyvale. If there is a feature you want added, you can head over to our [issues page](https://github.com/Computer-Aided-Validation-Laboratory/pyvale/issues) and use the "feature" tag to submit a request. We will then be able to discuss and provide feedback on your request.

## Contributing New Code

Code contributions should be associated with an issue or feature request. Please ensure you have submitted one of these before you contribute code.

## Licensing

Please be aware that all Pyvale code is under the [MIT Licencse](https://github.com/Computer-Aided-Validation-Laboratory/pyvale?tab=MIT-1-ov-file). It is your responsiblity to ensure that all contributions are compatible with this.

## Working on Pyvale

### Cloning the Repository

You can clone the pyvale repository at https://github.com/Computer-Aided-Validation-Laboratory/pyvale. 

Use the following command in the pyvale directory to install pyvale for development:
```shell
pip install pyvale -e .
```

We recommend installing Pyvale into a virtual environment of your choice as pyvale requires python 3.11.  If you need help setting up your virtual environment and installing pyvale head over to our [installation guide](https://computer-aided-validation-laboratory.github.io/pyvale/install/install.html).

### Make a Branch

Once you have cloned the repository and have an issue to work on, create a branch from dev to work on.
```shell
git switch dev
git switch -c branchname
```

### Docstrings

Docstrings are used to support making Pyvale documentation, as well as being a useful tool to help developers understand code that has been contributed. All contributions should include docstrings. These should include a short summary of what a function or class does, the parameters it takes, and what it returns.

See below for an example of how we structure our docstrings.
```python
def check_strain_files(strain_files: str | Path) -> list[str]:
    """
    Check for strain/deformation files in the given path and return their filenames.

    Parameters
    ----------
    strain_files : str or pathlib.Path
        Path or glob pattern pointing to the strain/deformation files.

    Returns
    -------
    list[str]
        A sorted list of filenames (not full paths) matching the input path/pattern.

    Raises
    ------
    FileNotFoundError
        If no files matching the given path or pattern are found.

    Examples
    --------
    >>> check_strain_files("data/strain_*.tif")
    ['strain_001.tif', 'strain_002.tif', 'strain_003.tif']
    """

    filenames = []

    # Find deformation image files
    files = sorted(glob.glob(str(strain_files)))
    if not files:
        raise FileNotFoundError(f"No DIC data found: {strain_files}")

    for file in files:
        filenames.append(os.path.basename(file))

    return filenames
```

### Testing

Pyvale uses [pytest](https://github.com/pytest-dev/pytest/). Please test your code before submitting a pull request. You can create your tests in the (pyvale/tests) directory. 

To run all tests, use the following command in the (pyvale) directory:
```shell
pytest tests
```
For more information on using pytest, refer to the [pytest docs](https://docs.pytest.org/en/stable/).


## Creating a Pull Request

Once you are satisfied with your contribution and have carried out tests, create a [pull request](https://github.com/Computer-Aided-Validation-Laboratory/pyvale/pulls) to merge into dev. 

The included pull request template will allow you to provide all the details needed for us to review your code. Once it has been reviewed it can be merged into dev where it will eventually be merged into main.