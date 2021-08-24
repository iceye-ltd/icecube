# Contributing

We accept Pull Requests!

To contribute, clone the repository and follow the instructions on the [Installation](installation.md) page.

### Pre-commit hooks

When creating a new branch, if you have used `inv setup` then pre-commit hooks should be enabled.
These check for formatting (`black` & `autoflake`) when using `git push`.

While it is possible to force push, it is recommended to pass all of them to push. In any case, a basic CI pipeline is in place for any push, which prevents a PR from merging if it fails.

### Testing

Tests are _not_ run at `push`, but they run in a CI pipeline. So, you can handle the tests the way you want to, but the tests must pass for your PR to be considered for approval :-)

It's a good habit to run them during the development process. The recommended way to run tests though is :

```
inv test
```

### Makefile options and tools

`tasks.py` offers many handy tools, such as the following:

- `inv -l`: Lists all the available `inv` options.
- `inv test`: Runs automated tests without the pre-commit hooks. This is handy to run tests without pushing code.
- `inv fmt`: Formats the source code, including tests using autoflake, black, and trim trailing whitespace.
