import marimo

__generated_with = "0.13.15"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # Bayesian Statistics

        > Marcos Duarte  
        > [Laboratory of Biomechanics and Motor Control](http://pesquisa.ufabc.edu.br/bmclab/)  
        > Federal University of ABC, Brazil
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        <h1>Table of Contents<span class="tocSkip"></span></h1>
        <div class="toc"><ul class="toc-item"><li><span><a href="#Definition" data-toc-modified-id="Definition-1"><span class="toc-item-num">1&nbsp;&nbsp;</span>Definition</a></span></li><li><span><a href="#Bayes'-theorem" data-toc-modified-id="Bayes'-theorem-2"><span class="toc-item-num">2&nbsp;&nbsp;</span>Bayes' theorem</a></span><ul class="toc-item"><li><span><a href="#Example:-Probability-of-having-a-disease-after-a-positive-result" data-toc-modified-id="Example:-Probability-of-having-a-disease-after-a-positive-result-2.1"><span class="toc-item-num">2.1&nbsp;&nbsp;</span>Example: Probability of having a disease after a positive result</a></span></li><li><span><a href="#Bayesian-Clinical-Diagnostic-Model" data-toc-modified-id="Bayesian-Clinical-Diagnostic-Model-2.2"><span class="toc-item-num">2.2&nbsp;&nbsp;</span>Bayesian Clinical Diagnostic Model</a></span></li></ul></li><li><span><a href="#References" data-toc-modified-id="References-3"><span class="toc-item-num">3&nbsp;&nbsp;</span>References</a></span></li></ul></div>
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Definition
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        From [Wikipedia](https://en.wikipedia.org/wiki/Bayesian_statistics):
    
        > Bayesian statistics is a theory in the field of statistics based on the Bayesian interpretation of probability where probability expresses a degree of belief in an event. The degree of belief may be based on prior knowledge about the event, such as the results of previous experiments, or on personal beliefs about the event.  
        This differs from a number of other interpretations of probability, such as the frequentist interpretation that views probability as the limit of the relative frequency of an event after a large number of trials.


        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Bayes' theorem

        Given two events$A$and$B$, the conditional probability of$A$given that$B$is true is expressed by the Bayes' theorem:$P(A \mid B) = \frac{P(B \mid A)P(A)}{P(B)}$where$P(B) \neq 0$.$P(A)$is the prior probability of$A$which expresses one's beliefs about$A$before evidence is taken into account;$P(B)$is the probability of the evidence$B$;$P(B∣A)$is the likelihood function, which can be interpreted as the probability of the evidence$B$given that$A$is true. The likelihood quantifies the extent to which the evidence$B$supports the proposition$A$;$P(A∣B)$is the posterior probability, the probability of the proposition$A$after taking the evidence$B$into account.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Example: Probability of having a disease after a positive result

        This example is taken from [https://youtu.be/R13BD8qKeTg](https://youtu.be/R13BD8qKeTg).  

        Suppose you test positive for a rare disease, with 0.1% of frequency of occurrence in the population.  
        You investigate more about the test for the disease and you found out that the employed test correctly identifies as positive 99% of the people who have the disease (true positives) and incorrectly identifies as positive 1% of the people who don't have the disease (false positives).  

        How certain you actually have the disease?  

        Let's use Bayes' theorem to answer this problem:

        The two events are: having the disease ($A$) and the positive result ($B$).  
        The question is: which is the probability,$P(A∣B)$, of actually having the disease given a positive result of the test for the disease.$P(A)$is the prior probability, the probability of having the disease before taking the test.$P(B∣A)$is the likelihood function, the probability of testing positive if you have the disease.$P(B)$is the probability of testing positive for the disease.

        With no more information about you and the disease, let's assume that$P(A)$is equal to the frequency of occurrence of the disease in the population,$0.001$.$P(B∣A)$is known,$0.99$.
        Concerning$P(B)$, we know there are two ways of testing positive for the disease: someone has the disease and the test result is positive and someone doesn't have the disease but the the test result is positive. The probability for the first case is$0.001 \times 0.99$and for the second case is$0.999 \times 0.01$. The total probability of testing positive is the sum of these two probabilities.  
        Them for$P(A∣B)$:$P(A\mid B) = \frac{P(B\mid A)P(A)}{P(B)} = \frac{0.99 \times 0.001}{0.001 \times 0.99 + 0.999 \times 0.01} = 0.09$So, there is a 9% chance that you actually have the disease. Given the accuracy of the test, 99%, and the fact that you tested positive, only 9% of chance seems surprising.
        """
    )
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
    mo.md(
        r"""
        Given that the probability that you have the disease is only 10%, you decide to do another test. In statistical terms, this second test should be independent of the first test.    
        Suppose you test positive again, now what is the probability that you actually have the disease?

        Now what is different is that we have more information about you and the disease, i.e., about the prior probability, P(A). Given that your first test was positive, we will use P(A) = 0.09016 and repeat the calculation.
        """
    )
    return


@app.cell
def _():
    _TP = 0.99
    _FP = 0.01
    _PBA = 0.99
    _PA = 0.09016
    _PB = _PA * _TP + (1 - _PA) * _FP
    _PAB = _PBA * _PA / _PB
    print('P(A|B) = {:.5f}'.format(_PAB))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        With two positive tests, there is a 91% probability that you actually have the disease.

        Note that even after two positive tests, this probability is still below the reported accuracy of the test (99%).
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Bayesian Clinical Diagnostic Model
        """
    )
    return


@app.cell
def _():
    from IPython.display import IFrame
    IFrame('https://kennis-research.shinyapps.io/Bayes-App/', width='100%', height=500)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## References

        - Cameron Davidson-Pilon (2015) [Bayesian Methods for Hackers: Probabilistic Programming and Bayesian Inference](https://camdavidsonpilon.github.io/Probabilistic-Programming-and-Bayesian-Methods-for-Hackers/).  
        """
    )
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
