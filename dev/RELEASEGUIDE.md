# Creating Releases

On the main page of the repository, click `releases` to the right of the files and create a new release.

Create a new tag for the release. Pyvale uses date based versioning in the format `vYYYY.M.N` where `Y` is the year,
`M` is the month, and `N` is an incremental value to differentiate multiple releases in a month. This number will also
be the title of your release.

Ensure you are targetting the correct branch - in our case, `main`.

Add a description of the release. The `generate release notes` option will get a list of all pull requests in this release.
