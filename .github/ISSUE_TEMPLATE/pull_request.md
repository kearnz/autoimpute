Name: **Pull Request**  
About: **Request to Merge your code into Autoimpute!**   

To make sure the pull request gets approved, you must follow the steps below:

1. Raise a bug report, create an issue, or request a new feature. Follow the templates for those to get started.
2. Once the authors review, develop the agreed upon solution. Do so by creating a feature branch.
3. When finished coding, ensure that you've written a unit test
        - Place the test in the tests/ directory. Choose the appropriate file, or ask us if you're unsure where to test.
        - Note that we use pytests, so you must prefix your function name with test_ to ensure it will run
4. Run the tests, ensuring that your test is successful and no other tests have broken.
5. Issue a pull request to merge your branch into the **dev branch**
        - We use Travis for CI, so pull requests to the dev branch run all appropriate integration testing.
        - If your tests succeeds, the authors will rebase and merge your pull request.
6. Once merged to dev, we'll take care of merging to master (via a version bump or hotfix depending on what you merged)
