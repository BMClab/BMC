# /// script
# requires-python = ">=3.11"
# dependencies = ["marimo>=0.23.9"]
# ///

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Version control with Git and GitHub

    > Marcos Duarte, Renato Naville Watanabe<br>
    > [Laboratory of Biomechanics and Motor Control](https://bmclab.pesquisa.ufabc.edu.br/)<br>
    > Federal University of ABC, Brazil
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## How to use this guide

    You corrected an equation in a notebook last week. Today you need to explain what changed and recover the earlier version. Git records that history locally; GitHub can host a shared copy. This guide teaches the small workflow needed for text-based Marimo notebooks.

    Read the examples in order, predict each result, and then run the cell. Try the distributed Exercises before opening their worked solutions. Finish with the checkpoint questions and independent Problems; their panels contain final answers only.

    **Challenge 0.** You edit a file after staging it, then commit. Does the commit contain the latest edit or the staged snapshot?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Python setup

    Only Marimo is required to read this guide. To practice, install [Git](https://git-scm.com/downloads) and use a disposable practice repository in a terminal. The commands below are displayed as text; this notebook does not execute Git commands or publish anything.
    """)
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Repository, working tree, and staging area

    A **repository** stores commits and their history. Your **working tree** contains the files you edit. The **index**, or staging area, holds the snapshot selected for the next commit. A **branch** is a movable reference to a line of history; a **clone** is a local repository copy; a GitHub **fork** is a repository under another owner. A fork and a branch are different things.

    ![Git local operations](https://raw.githubusercontent.com/BMClab/BMC/master/images/GitLocalOperations.png)

    Git is distributed: a local commit does not need GitHub or Internet access. A remote copy helps collaboration and backup, but Git does not automatically capture untracked data or unsaved edits.

    ```sh
    git status
    git diff
    git diff --cached
    ```

    `status` summarizes files. `diff` shows unstaged changes; `diff --cached` shows the staged changes relative to the last commit.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Exercise 1.** You stage a corrected equation, then change its caption. What does the next ordinary `git commit` include?

    /// details | Solution to Exercise 1
    It includes the staged equation correction. The later caption change remains unstaged until you add it too. Use both diff commands to see the two snapshots separately.
    ///
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Start and record a small change

    For a new practice folder use `git init`. To obtain this collection instead, use `git clone https://github.com/BMClab/BMC.git`, then `cd BMC`. A clone already has a repository; do not initialize another inside it.

    Set your author identity in your practice repository. These commands use repository-local configuration; adding `--global` changes the default for your other repositories too. An email associated with your GitHub account, including its private noreply address, can link commits to that account.

    ```sh
    git config user.name "Your Name"
    git config user.email "your-address@example.com"
    git add notebook.py
    git diff --cached
    git commit -m "Correct units in the walking example"
    git log --oneline
    ```

    Run the notebook before recording the change. Stage explicit paths so temporary outputs or unrelated edits do not enter the commit. A `.gitignore` rule excludes matching untracked artifacts; it does not stop tracking a file already in history. `git rm filename` removes and stages a tracked file; deleting it manually and then staging the deletion is also valid.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Exercise 2.** A generated HTML export and a source notebook both changed. How would you record only the source correction?

    /// details | Solution to Exercise 2
    Inspect `git status`, run `git add notebook.py`, review `git diff --cached`, and commit. Leave the generated export unstaged. The source and reproducible environment normally explain the result better than a pile of temporary exports.
    ///
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Synchronize deliberately

    A remote is a named repository URL. `origin` is the usual name created by cloning. Inspect it before sending changes:

    ```sh
    git remote -v
    git branch --show-current
    git fetch origin
    git pull --ff-only
    ```

    `fetch` downloads history without integrating it into your working branch. `pull --ff-only` fetches and advances that branch only when no merge is needed. If histories diverge, inspect the changes and choose a merge or other agreed workflow; repeating the command will not resolve a conflict. A merge combines histories, and conflicting edits need a human decision followed by a notebook run.

    When you intend to publish and have write access, `git push origin YOUR_BRANCH` sends local commits to that branch. Replace `YOUR_BRANCH` with the actual branch name; this repository uses `master`, while other projects may use `main` or another name. GitHub HTTPS authentication uses a credential manager or token, not your account password; SSH uses a registered key. See [GitHub authentication](https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/about-authentication-to-github).

    If you fork a project, your fork is usually `origin`. You can name the original repository `upstream`, fetch it, and integrate its changes intentionally. Cloning a public repository does not give you permission to push to it.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Exercise 3.** Your local notebook is committed while the network is unavailable. What is saved, and what remains to happen before a collaborator can fetch it?

    /// details | Solution to Exercise 3
    The commit is saved in your local repository. It must be pushed to a remote that your collaborator can access. GitHub is not involved in making the local commit.
    ///
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Recover and review

    `git log --oneline` finds commits; `git show COMMIT` inspects one. `git restore --staged notebook.py` removes a change from the staging area while keeping the working file. `git restore notebook.py` instead discards unstaged changes in that file, so inspect the diff first.

    `git revert COMMIT` records a new commit that reverses the changes of the named commit. It does **not** cancel every later commit. This is useful when a published correction itself needs correcting. For experimental work, `git switch -c experiment` starts a new branch; branch workflows are optional for the small exercises here.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Exercise 4.** You want to unstage a notebook but keep your edits. Which restore command fits, and which would risk discarding them?

    /// details | Solution to Exercise 4
    Use `git restore --staged notebook.py`. Without `--staged`, `git restore notebook.py` restores the working file from the index and discards its unstaged edits. The destination matters.
    ///
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Checkpoint questions

    1. Is a commit automatically a push?
    2. What snapshot does `git diff --cached` inspect?
    3. Is a fork the same as a branch?

    /// details | Checkpoint answers
    1. No.
    2. The index relative to the last commit.
    3. No; a fork is another repository.
    ///

    ## Problems

    1. Name the command to inspect unstaged changes.
    2. Name the command to unstage `guide.py` while preserving its working copy.
    3. Does `git revert ABC123` undo every commit after ABC123?
    4. What happens when `git pull --ff-only` encounters diverging histories?

    /// details | Final answers
    1. `git diff`.
    2. `git restore --staged guide.py`.
    3. No; it reverses the changes introduced by ABC123 in a new commit.
    4. It refuses the integration; the histories need an explicit decision.
    ///

    ## Go deeper

    [Pro Git](https://git-scm.com/book/en/v2) — work through recording changes, remotes, and undoing changes when you need more detail.

    ### Video lectures on the Internet

    [MIT Missing Semester: Version Control](https://missing.csail.mit.edu/2020/version-control/) — an embedded lecture on the history graph and the meaning of everyday commands.

    ## References

    - Scott Chacon and Ben Straub, [Pro Git](https://git-scm.com/book/en/v2). [Git reference](https://git-scm.com/docs). [GitHub authentication](https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/about-authentication-to-github).
    """)
    return


if __name__ == "__main__":
    app.run()
