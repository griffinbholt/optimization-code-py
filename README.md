# optimization-code-py

[![Python package](https://github.com/griffinbholt/optimization-code-py/actions/workflows/python-package.yml/badge.svg)](https://github.com/griffinbholt/optimization-code-py/actions/workflows/python-package.yml)

*Original Julia Code by: Mykel Kochenderfer and Tim Wheeler*

*Python Versions by: Griffin Holt*

Python versions of all typeset code blocks from the book, [Algorithms for Optimization](https://algorithmsbook.com/optimization/).

I share this content in the hopes that it helps you and makes the decision making algorithms more approachable and accessible (especially to those not as familiar with Julia). Thank you for reading!

If you encounter any issues or have pressing comments, please [file an issue](https://github.com/griffinbholt/optimization-code-py/issues/new/choose). (There are likely to still be bugs as I have not finished testing all of the classes and functions.)

## Progress Update: (13 Mar 2024)

| Chapter(s) | Written | Tested | Notes |
|--:|:--|:--|:--|
|  1 | N/A | N/A | No code blocks in this chapter |
|  2 | ▌▌▌▌▌▌▌▌▌▌ 100% | ▌▌▌▌▌▌▌▌▌▌ 100% | **Ready for use** |
|  3 | ▌▌▌▌▌▌▌▌▌▌ 100% | ▌▌▌▌▌▌▌▌▌▌ 100% | **Ready for use** |
|  4 | ▌▌▌▌▌▌▌▌▌▌ 100% | 0% | Needs to be tested |
|  5 | ▌▌▌▌▌▌▌▌▌▌ 100% | 0% | Needs to be tested |
|  6 | ▌▌▌▌▌▌▌▌▌▌ 100% | 0% | Needs to be tested |
|  7 | ▌▌▌▌▌▌▌▌▌▌ 100% | 0% | Needs to be tested |
|  8 | ▌▌▌▌▌▌▌▌▌▌ 100% | 0% | Needs to be tested |
|  9 | ▌▌▌▌▌▌▌▌▌▌ 100%  | 0% | Needs to be tested |
| 10 | ▌▌▌▌▌▌▌▌▌▌ 100%  | 0% | Needs to be tested |
| 11 | ▌▌▌▌▌▌▌▌▌▌ 100%  | 0% | Needs to be tested |
| 12 | ▌▌▌▌▌▌▌▌▌▌ 100%  | 0% | Needs to be tested |
| 13 | ▌▌▌▌▌▌▌▌▌▌ 100%  | 0% | Needs to be tested |
| 14 | ▌▌▌▌▌▌▌▌▌▌ 100%  | 0% | Needs to be tested |
| 15 | ▌▌▌▌▌▌▌▌▌▌ 100%  | 0% | Needs to be tested |
| 16 | ▌▌▌▌▌▌▌▌▌▌ 100%  | 0% | Needs to be tested |
| 17 | N/A | N/A | No code blocks in this chapter |
| 18 | ▌▌▌▌▌▌▌▌▌▌ 100%  | 0% | Needs to be tested |
| 19 | ▌▌▌▌▌▌▌▌▌▌ 100%  | 0% | Needs to be tested |
| 20 | 0% | 0% | Need to figure out replacement library for `ExprRules.jl` |
| 21 | ▌▌▌▌▌▌▌▌▌▌ 100%  | ▌▌▌▌▌▌▌▌▌▌ 100% | **Ready for use** |

I have also written code for pertinent figures, examples, exercises through Chapter 4.

I have also written code for test functions (`TestFunctions.py`), convenience functions (`convenience.py`),
and probability distributions with `fit` functionality (`Distributions.py`).

<!-- TODO - I need to go through and check that all functions have proper parameter
and return signatures. -->

<!-- TODO - I need to go through all of the def(...)... one line functions and make sure that parameters are passed through so they persist. -->

<!-- TODO - Suppress the Deprecated Warnings in pytest: https://docs.pytest.org/en/stable/how-to/capture-warnings.html -->