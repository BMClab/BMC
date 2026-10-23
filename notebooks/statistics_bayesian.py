# /// script
# requires-python = ">=3.11"
# dependencies = ["marimo>=0.23.9", "ipython"]
# ///

import marimo

__generated_with = "0.24.2"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Bayesian Statistics

    > Marcos Duarte, Renato Naville Watanabe<br>
    > [Laboratory of Biomechanics and Motor Control](https://bmclab.pesquisa.ufabc.edu.br/)<br>
    > Federal University of ABC, Brazil
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## How to use this guide

    A positive result sounds decisive until you ask how common the condition was before testing. This guide keeps the arithmetic visible: count the two routes to a positive result, apply Bayes’ theorem, then ask whether another test really contributes independent evidence. The disease example is a hypothetical probability model.

    Read the examples in order, predict each result, and then run the cell. Try the distributed Exercises before opening their worked solutions. Finish with the checkpoint questions and independent Problems; their panels contain final answers only.

    **Challenge 0.** Imagine 100,000 people and a condition present in 0.1% of them. A test detects 99% of cases but flags 1% of non-cases. Will most positive results be true cases?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Python setup

    Only Python arithmetic and Marimo are required for the calculations. The linked diagnostic app is optional and requires an Internet connection; the probability examples here remain self-contained.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Definition
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    From [Wikipedia](https://en.wikipedia.org/wiki/Bayesian_statistics):

    > Bayesian statistics is a theory in the field of statistics based on the Bayesian interpretation of probability where probability expresses a degree of belief in an event. The degree of belief may be based on prior knowledge about the event, such as the results of previous experiments, or on personal beliefs about the event.<br>
    This differs from a number of other interpretations of probability, such as the frequentist interpretation that views probability as the limit of the relative frequency of an event after a large number of trials.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Bayes' theorem

    Given two events $A$ and $B$, the conditional probability of $A$ given that $B$ is true is expressed by the Bayes' theorem:

    $$
    P(A \mid B) = \frac{P(B \mid A)P(A)}{P(B)}
    $$

    where $P(B) \neq 0$.

    $P(A)$ is the prior probability of $A$ which expresses one's beliefs about $A$ before evidence is taken into account;<br>
    $P(B)$ is the probability of the evidence $B$;<br>
    $P(B∣A)$ is the likelihood function, which can be interpreted as the probability of the evidence $B$ given that $A$ is true. The likelihood quantifies the extent to which the evidence $B$ supports the proposition $A$;<br>
    $P(A∣B)$ is the posterior probability, the probability of the proposition $A$ after taking the evidence $B$ into account.
    """)
    return




@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Exercise 1.** Are $P(A\mid B)$ and $P(B\mid A)$ interchangeable? Use the screening example to name each.

    /// details | Solution to Exercise 1
    No. $P(B\mid A)$ is sensitivity: the chance of a positive result among cases. $P(A\mid B)$ is the proportion of positives who are cases. Bayes’ denominator counts **all** ways of obtaining a positive result, so prevalence matters.
    ///
    """)
    return
@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Example: Probability of having a disease after a positive result

    This example is taken from [https://youtu.be/R13BD8qKeTg](https://youtu.be/R13BD8qKeTg).

    Suppose you test positive for a rare disease, with 0.1% of frequency of occurrence in the population.<br>
    You investigate more about the test for the disease and you found out that the employed test correctly identifies as positive 99% of the people who have the disease (true positives) and incorrectly identifies as positive 1% of the people who don't have the disease (false positives).

    How certain you actually have the disease?

    Let's use Bayes' theorem to answer this problem:

    The two events are: having the disease ($A$) and the positive result ($B$).<br>
    The question is: which is the probability, $P(A∣B)$, of actually having the disease given a positive result of the test for the disease.<br>
    $P(A)$ is the prior probability, the probability of having the disease before taking the test.<br>
    $P(B∣A)$ is the likelihood function, the probability of testing positive if you have the disease.<br>
    $P(B)$ is the probability of testing positive for the disease.

    With no more information about you and the disease, let's assume that $P(A)$ is equal to the frequency of occurrence of the disease in the population, $0.001$.
    $P(B∣A)$ is known, $0.99$.
    Concerning $P(B)$, we know there are two ways of testing positive for the disease: someone has the disease and the test result is positive and someone doesn't have the disease but the the test result is positive. The probability for the first case is $0.001 \times 0.99$ and for the second case is $0.999 \times 0.01$. The total probability of testing positive is the sum of these two probabilities.<br>
    Them for $P(A∣B)$:

    $$
    P(A\mid B) = \frac{P(B\mid A)P(A)}{P(B)} = \frac{0.99 \times 0.001}{0.001 \times 0.99 + 0.999 \times 0.01} = 0.09
    $$

    So, there is a 9% chance that you actually have the disease. Given the sensitivity and specificity of the test, both 99%, and the fact that you tested positive, only 9% of chance seems surprising.
    """)
    return


@app.cell
def _():
    FO = 0.001
    _TP = 0.99
    _FP = 0.01
    _PBA = 0.99
    _PA = FO
    _PB = _PA * _TP + (1 - _PA) * _FP
    _PAB = _PBA * _PA / _PB
    print('P(A|B) = {:.5f}'.format(_PAB))
    return




@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Exercise 2.** In 100,000 people, calculate true and false positives under the example assumptions. Recover the posterior without a formula.

    /// details | Solution to Exercise 2
    There are 100 cases and 99,900 non-cases. Expected true positives: $0.99(100)=99$; false positives: $0.01(99{,}900)=999$. Thus 99 of 1,098 positives are cases: $99/1098=0.09016$, about 9.02%. These are expected counts, not observations.
    ///
    """)
    return
@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    In this hypothetical model, the posterior after one positive result is about 9.02%. Suppose a second test has the same sensitivity and false-positive rate. Multiplying their likelihoods requires **conditional independence given disease status**, both among cases and among non-cases. Repeating the same test may share errors and need not satisfy this assumption.<br>
    Suppose you test positive again, now what is the probability that you actually have the disease?

    Now what is different is that we have more information about you and the disease, i.e., about the prior probability, P(A). Given that your first test was positive, we will use P(A) = 0.09016 and repeat the calculation.
    """)
    return


@app.cell
def _():
    _TP = 0.99
    _FP = 0.01
    _PBA = 0.99
    _PA = (0.99 * 0.001) / (0.99 * 0.001 + 0.01 * 0.999)
    _PB = _PA * _TP + (1 - _PA) * _FP
    _PAB = _PBA * _PA / _PB
    print('P(A|B) = {:.5f}'.format(_PAB))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    With two positive tests, there is a 91% probability that you actually have the disease.

    Note that even after two positive tests, this probability is still below the sensitivity and specificity of the test (99%).
    """)
    return




@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Exercise 3.** Suppose the second test merely copies the first result. Does a second positive change the posterior?

    /// details | Solution to Exercise 3
    No. Once the first positive is known, a copied result is certain in both groups. Its likelihood ratio is one, so the posterior remains about 9.02%. Treating the copy as independent would count the same evidence twice.
    ///
    """)
    return
@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Bayesian Clinical Diagnostic Model
    """)
    return


@app.cell
def _():
    from IPython.display import IFrame
    IFrame('https://kennis-research.shinyapps.io/Bayes-App/', width='100%', height=500)
    return




@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Checkpoint questions

    1. What is the prior?
    2. Why is the denominator not just sensitivity?
    3. Which independence assumption permits sequential updating with the same likelihoods?

    /// details | Checkpoint answers
    1. Probability before the new evidence.
    2. It includes true and false positives.
    3. Conditional independence of test results within each disease-status group.
    ///

    ## Problems

    1. Prevalence 1%, sensitivity 90%, specificity 95%: find the probability of the condition after one positive result.
    2. In the original 0.1%-prevalence example, give the posterior after two conditionally independent positives.
    3. Can a higher prior change the posterior when the test itself is unchanged?

    /// details | Final answers
    1. 15.38%.
    2. 90.75%.
    3. Yes; a higher prior gives a higher posterior for the same positive likelihood ratio.
    ///

    ## Go deeper

    [MIT probability and statistics](https://ocw.mit.edu/courses/18-05-introduction-to-probability-and-statistics-spring-2022/) — the Bayesian inference readings extend this discrete example.

    ### Video lectures on the Internet

    [Bayes’ theorem, 3Blue1Brown](https://www.youtube.com/watch?v=HZGCoVF3YvM) — a geometric explanation of how prior information changes the meaning of evidence.
    """)
    return
@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## References

    - Cameron Davidson-Pilon (2015) [Bayesian Methods for Hackers: Probabilistic Programming and Bayesian Inference](https://camdavidsonpilon.github.io/Probabilistic-Programming-and-Bayesian-Methods-for-Hackers/).
    """)
    return


@app.cell
def _():
    import marimo as mo

    return (mo,)


if __name__ == "__main__":
    app.run()
