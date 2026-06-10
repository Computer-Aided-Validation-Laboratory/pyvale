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

## AI Usage

The Pyvale project has strict rules for AI usage:

- **All AI usage in any form must be disclosed.** You must state the tool you used (e.g. Claude Code, Cursor, Amp) along with the extent that the work was AI-assisted.

- **The human-in-the-loop must fully understand all code.** If you can't explain what your changes do and how they interact with the greater system without the aid of AI tools, do not contribute to this project.

- **Issues and discussions can use AI assistance but must have a full human-in-the-loop.** This means that any content generated with AI must have been reviewed and edited_ by a human before submission. AI is very good at being overly verbose and including noise that distracts from the main point. Humans must do their research and trim this down.

- **No AI-generated media is allowed (art, images, videos, audio, etc.).** Text and code are the only acceptable AI-generated content, per the other rules in this policy.

- **Bad AI drivers will be denounced** People who produce bad contributions that are clearly AI (slop) will be added to our denouncement list. This list will be used to block all future contributions. 

These rules apply only to outside contributions to Pyvale. Maintainers are exempt from these rules and may use AI tools at their discretion; they've proven themselves trustworthy to apply good judgment.

### There are Humans Here

Please remember that Pyvale is maintained by humans.

Every discussion, issue, and pull request is read and reviewed by humans (and sometimes machines, too). It is a boundary point at which people interact with each other and the work done. It is rude and disrespectful to approach this boundary with low-effort, unqualified work, since it puts the burden of validation on the maintainer.

In a perfect world, AI would produce high-quality, accurate work every time. But today, that reality depends on the driver of the AI. And today, most drivers of AI are just not good enough. So, until either the people get better, the AI gets better, or both, we have to have strict rules to protect maintainers.

### AI is Welcome Here

As a project, we welcome AI as a tool!

**Our reason for the strict AI policy is not due to an anti-AI stance**, but instead due to the number of highly unqualified people using AI. It's the people, not the tools, that are the problem.

I include this section to be transparent about the project's usage about AI for people who may disagree with it, and to address the misconception that this policy is anti-AI in nature.

_This policy has been adapted from the [Ghostty AI policy](https://github.com/ghostty-org/ghostty/blob/main/AI_POLICY.md?plain=1). Credit goes to the [Ghostty](https://github.com/ghostty-org/ghostty/tree/main) team._

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

### Developer Guidance

Much of Pyvale is written in Python. See our [Pyvale Developer Guide](https://github.com/Computer-Aided-Validation-Laboratory/pyvale/blob/main/designspec/README.md) for guidance on how to write Python code that best aligns with Pyvale's code values.

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