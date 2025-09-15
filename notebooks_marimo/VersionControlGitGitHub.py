import marimo

__generated_with = "0.13.15"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # Version control with Git and GitHub

        > Marcos Duarte  
        > [Laboratory of Biomechanics and Motor Control](http://pesquisa.ufabc.edu.br/bmclab)   
        > Federal University of ABC, Brazil
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        <h1>Contents<span class="tocSkip"></span></h1>
        <div class="toc"><ul class="toc-item"><li><span><a href="#Basic-glossary-of-version-control" data-toc-modified-id="Basic-glossary-of-version-control-1"><span class="toc-item-num">1&nbsp;&nbsp;</span>Basic glossary of version control</a></span></li><li><span><a href="#Git" data-toc-modified-id="Git-2"><span class="toc-item-num">2&nbsp;&nbsp;</span>Git</a></span><ul class="toc-item"><li><span><a href="#Installing-Git" data-toc-modified-id="Installing-Git-2.1"><span class="toc-item-num">2.1&nbsp;&nbsp;</span>Installing Git</a></span></li><li><span><a href="#Configuring-Git" data-toc-modified-id="Configuring-Git-2.2"><span class="toc-item-num">2.2&nbsp;&nbsp;</span>Configuring Git</a></span></li><li><span><a href="#Initialize-a-Git-repository" data-toc-modified-id="Initialize-a-Git-repository-2.3"><span class="toc-item-num">2.3&nbsp;&nbsp;</span>Initialize a Git repository</a></span></li></ul></li><li><span><a href="#Checkout-a-repository" data-toc-modified-id="Checkout-a-repository-3"><span class="toc-item-num">3&nbsp;&nbsp;</span>Checkout a repository</a></span></li><li><span><a href="#The-working-directory" data-toc-modified-id="The-working-directory-4"><span class="toc-item-num">4&nbsp;&nbsp;</span>The working directory</a></span></li><li><span><a href="#Add-and-commit-changes" data-toc-modified-id="Add-and-commit-changes-5"><span class="toc-item-num">5&nbsp;&nbsp;</span>Add and commit changes</a></span></li><li><span><a href="#Push-changes" data-toc-modified-id="Push-changes-6"><span class="toc-item-num">6&nbsp;&nbsp;</span>Push changes</a></span></li><li><span><a href="#Update-and-merge-changes" data-toc-modified-id="Update-and-merge-changes-7"><span class="toc-item-num">7&nbsp;&nbsp;</span>Update and merge changes</a></span></li><li><span><a href="#Remove-or-delete-files" data-toc-modified-id="Remove-or-delete-files-8"><span class="toc-item-num">8&nbsp;&nbsp;</span>Remove or delete files</a></span></li><li><span><a href="#Syncing-a-fork" data-toc-modified-id="Syncing-a-fork-9"><span class="toc-item-num">9&nbsp;&nbsp;</span>Syncing a fork</a></span></li><li><span><a href="#Git-Commands-Cheat-Sheet" data-toc-modified-id="Git-Commands-Cheat-Sheet-10"><span class="toc-item-num">10&nbsp;&nbsp;</span>Git Commands Cheat Sheet</a></span></li><li><span><a href="#Further-reading" data-toc-modified-id="Further-reading-11"><span class="toc-item-num">11&nbsp;&nbsp;</span>Further reading</a></span></li><li><span><a href="#Video-lectures-on-the-Internet" data-toc-modified-id="Video-lectures-on-the-Internet-12"><span class="toc-item-num">12&nbsp;&nbsp;</span>Video lectures on the Internet</a></span></li><li><span><a href="#References" data-toc-modified-id="References-13"><span class="toc-item-num">13&nbsp;&nbsp;</span>References</a></span></li></ul></div>
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Version control has become a powerful tool to back up data, keep your work organized, and collaborate with others, very useful in the academic life.

        > Revision control, also known as version control and source control, is the management of changes to documents, computer programs, large web sites, etc.  
        Changes are usually identified by a number or letter code, termed the "revision number", "revision level", or simply "revision". For example, an initial set of files is "revision 1". When the first change is made, the resulting set is "revision 2", and so on. Each revision is associated with a timestamp and the person making the change. Revisions can be compared, restored, and with some types of files, merged.   
        Version control systems (VCS) most commonly run as stand-alone applications, but revision control is also embedded in various types of software such as word processors and spreadsheets. Revision control allows for the ability to revert a document to a previous revision, which is critical for allowing editors to track each other's edits and correct mistakes (even if you work alone, version control is useful for that).
        > From [http://en.wikipedia.org/wiki/Revision_control](http://en.wikipedia.org/wiki/Revision_control).
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Basic glossary of version control

        - **Repository**: the core of VCS, a data structure usually stored on a server that contains a set of files and directories, a historical record of changes in the repository, a set of commit objects, and a set of references to commit objects (called heads).
        - **Branch, fork, clone**: a branch in a repository, or a fork or clone of an entire repository, are different forms of copies of a repository. The main branch in a repository is called master or trunk. You can work on this copy and then merge it in to the master branch/repository.
        - **Checkout**: to checkout the repository is to obtain a local working copy of the files. Changes can be made to these local files and files can be added, removed and updated.
        - **Commit**: to commit a file to the repository means that the changes to the local files are saved to the repository (committed).
        - **Pull, push**: we can pull and push changesets between different repositories. For example, between a local copy of the repository to a central online repository.
        - **Diff**: a diff is the difference in changes between two commits, or saved changes.
        - **[Git](https://github.com/)** is a free and open source distributed version control software. Git is a very popular VCS nowadays.   
        - **[GitHub](https://github.com/)** is a web-based hosting service for software development projects that use the Git VCS. GitHub offers both paid plans for private repositories, and free accounts for open source projects. GitHub is also very popular.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Git

        Let's see how to use Git for version control. For more on that, see this [Git guide](http://rogerdudler.github.io/git-guide/), the [Pro Git Book](http://git-scm.com/book), and the [GitHub help](https://docs.github.com/en/github/getting-started-with-github).

        "Git has three main states that your files can reside in: committed, modified, and staged.   
        Committed means that the data is safely stored in your local database. Modified means that you have changed the file but have not committed it to your database yet. Staged means that you have marked a modified file in its current version to go into your next commit snapshot.   
        This leads us to the three main sections of a Git project: the Git directory, the working directory, and the staging area." ([Pro Git Book](http://git-scm.com/book))

        <div class='center-align'><figure><img src="./../images/GitLocalOperations.png"/><figcaption><i>Working directory, staging area, and Git directory ([Pro Git Book](http://git-scm.com/book)).</i></figcaption></figure></div> 

        The local repository consists of three "trees" maintained by Git:

        1. Working Directory holds the actual files.   
        2. Index acts as a staging area.   
        3. HEAD points to the last commit you've made.

        The basic Git workflow is ([Pro Git Book](http://git-scm.com/book)):

        - Modify files in the working directory.
        - Stage the files, adding snapshots of them to the staging area.
        - Do a commit, which takes the files as they are in the staging area and stores that snapshot permanently to the Git directory.

        <div><center><figure><img src="./../images/GitFileLifecycle.png"/><figcaption><i>Figure. Lifecycle of a file using Git ([Pro Git Book](http://git-scm.com/book)).</i></figcaption></figure></center></div> 
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Installing Git

        See [http://git-scm.com/downloads](http://git-scm.com/downloads) for detailed instructions on how to install Git, in short:   

        On MS Windows, install it using [http://git-scm.com/download/win](http://git-scm.com/download/win).

        On Linux (Debian/Ubuntu):    

        >`$sudo apt-get install git`  

        On Mac OS X, there are different ways:   

        - Use a Git installer, such as [http://git-scm.com/download/mac](http://git-scm.com/download/mac).
        - With [Homebrew](http://brew.sh/), use the command line:    

        >`$brew install git`    

        - With [MacPorts](http://www.macports.org/), use the command line:     

        >`$sudo port install git`

        You can also install a graphical user interface (GUI) for Git: [GUI Clients](http://git-scm.com/downloads/guis).    
        In MS Windows, if you installed the official Git (cited above), you already installed a GUI (Git GUI), look at the Git folder. Anyway, if you are in MS Windows or Mac OS X and have a GitHub account, you may want to consider to use the [GitHub GUI](http://git-scm.com/downloads/guis) GUI because integrates easily with your GitHub account.

        Let's see now a short tutorial on how to use Git and GitHub for version control with command lines (if you plan to work with a GUI client, just the concepts are important). 

        After you installed Git, you can check its version using a terminal window or the command `!` in the IPython Notebook to access the system shell.   
        In MS Windows, however, if you installed Git with the recommended default options, the commands below will not work and the only terminal window (command prompt window) that works is the `Git Bash` that was installed with Git. So, open `Git Bash` and run the commands below (always without the `!`).
        """
    )
    return


app._unparsable_cell(
    r"""
    !git --version
    """,
    name="_"
)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        And if you type `git` you get a list of the most common commands in Git:
        """
    )
    return


app._unparsable_cell(
    r"""
    !git
    """,
    name="_"
)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Configuring Git

        After installation, we need to configure Git (this is only needed once and it will be used when we commit changes to the repository).   
        Let's do that using the following command lines:
        """
    )
    return


app._unparsable_cell(
    r"""
    !git config --global user.name \"your name here\"
    !git config --global user.email \"your_email@example.com\"
    """,
    name="_"
)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Your email address for Git should be the same one associated with your GitHub account in case you plan to have a repository there.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Initialize a Git repository

        To initialize a local Git repository in the current directory:
        """
    )
    return


app._unparsable_cell(
    r"""
    !git init
    """,
    name="_"
)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        You would do that in case you want starting to track an existing local project in Git.   
        You can also specify a new local repository, with the command `git init <repository-name>`.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Checkout a repository

        To clone a remote repository, we can use two different protocols to transfer data: `HTTPS` and `SSH`. `HTTPS` and `SSH` are cryptographic network protocols for secure data communication, remote command-line login, remote command execution, and other secure network services between two networked computers that connects, via a secure channel over an insecure network. `HTTPS ` is simpler to setup and `SSH` requires a keypair generated on your computer and attached to your GitHub account. See this [GitHub help](https://help.github.com/articles/which-remote-url-should-i-use) to decide which one to use.   

        For instance, this is the command to clone repo of this notebook using `HTTPS`:
        """
    )
    return


app._unparsable_cell(
    r"""
    !git https://github.com/BMClab/BMC.git
    """,
    name="_"
)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        And using `SSH` (after you created your keypair and registered into you GitHub account):
        """
    )
    return


app._unparsable_cell(
    r"""
    !git clone git@github.com:BMClab/BMC.git
    """,
    name="_"
)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        The local repository will be created inside your current directory.   
        You can change the current directory to clone from there and there is no need to create a folder with the name of the repo you are cloning; Git will do that for you.   
        And you should not have your local repo inside a Dropbox folder because Dropbox can generate conflict files.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## The working directory

        The Git commands will only work once you are in your working directory of your local repository.

        Use the command `cd` to change you current directory. In Linux or Mac OS X (change the directory to your case):
        """
    )
    return


@app.cell
def _(DATA, GitHub, cd, mnt):
    cd /mnt/DATA/GitHub
    return


@app.cell
def _(pwd):
    pwd
    return


@app.cell
def _(ls):
    ls
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Add and commit changes

        You can propose changes (add it to the Index) using the command `add`.  
        For instance, let's change the README.md file, and commit it:
        """
    )
    return


app._unparsable_cell(
    r"""
    !git add README.md
    """,
    name="_"
)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        The command `add` is a multipurpose command; allows to track files, stage files, and mark merge conflicted files as resolved.  
        You can add everything in the current directory using the command "`git add -A`".  
  
        To commit this change:
        """
    )
    return


app._unparsable_cell(
    r"""
    !git commit -m \"Commit message\"
    """,
    name="_"
)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Which commits everything in your staging area and uses inline commit message.

        If the file to commit is not new, only changed, you can skip the `add` command using the command `commit -a` to automatically stages every currently **tracked** file and commits them:
        """
    )
    return


app._unparsable_cell(
    r"""
    !git commit –a –m \"Commit message\"
    """,
    name="_"
)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        If you created/added a new file (untracked so far) you still need to add them to your staging area with the command `add`.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Push changes

        Now the file is committed to the HEAD, but not in the remote repository yet.   

        To send this changes to the remote repository, execute (substitute `master` by the branch to push the change to if it's not the same repo you cloned from):
        """
    )
    return


app._unparsable_cell(
    r"""
    !git push origin master
    """,
    name="_"
)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        You should do that in a regular terminal window because you may be prompted to enter your username and password for your GitHub account in case you didn't store the username and password in your OS (if you are using `HTTPS`).   
        If you are using `SSH`, by default you will not be prompted to enter your credentials (because you created your keypair and registered into you GitHub account).

        Anyway, if you are committing to my repo, this will nor work for you because, I hope, you don't have my credentials.

        You can create a branch (fork) of the BMC repository (go to its website and use the `Fork` button in the upper-right corner) or create your own repo to experiment with it.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Update and merge changes

        To update the local repository to the newest commit, execute in the working directory (this will fetch and merge remote changes):
        """
    )
    return


app._unparsable_cell(
    r"""
    !git pull
    """,
    name="_"
)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        To merge another branch into your active branch (e.g. master), use `git merge <branch>`.

        These two last commands tries to auto-merge changes.   
        This might not be possible because of conflicts between the different branches.   
        If that is the case, we will have to merge those conflicts manually by editing the files shown by git.    
        After changing, we need to mark them as merged with `git add <filename>` or `git add .`.   
        Before merging changes, we can preview them by using `git diff <source_branch> <target_branch>`.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Remove or delete files

        You should not manually remove or delete a file inside your repository, for that use the command `rm`:
        """
    )
    return


app._unparsable_cell(
    r"""
    !git rm -help
    """,
    name="_"
)


app._unparsable_cell(
    r"""
    !git rm <filename>
    """,
    name="_"
)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Syncing a fork

        To sync a fork of a repository to keep it up-to-date with the upstream repository (for example, if you forked the BMClab/BMC repo), according to the [GitHub Help](https://help.github.com/articles/syncing-a-fork/), you have to first configure a remote for a fork and then fetch the commits (the changes) from the upstream repository.  
        Follow these steps:

        1. Open Terminal (for Mac and Linux users) or the command line (for Windows users).  
        2. Change the current working directory to your local project.
        3. List the current configured remote repository for your fork:
        ```
        git remote -v
        ```
        4. Specify a new remote upstream repository that will be synced with the fork, type:
        ```
        git remote add upstream https://github.com/BMClab/BMC.git
        ```
        5. Verify the new upstream repository you've specified for your fork:
        ```
        git remote -v
        ```
        6. Fetch the branches and their respective commits from the upstream repository. Commits to master will be stored in a local branch, upstream/master:
        ```
        git fetch upstream
        ```
        7. Check out your fork's local master branch:
        ```
        git checkout master
        ```
        8. Merge the changes from upstream/master into your local master branch. This brings your fork's master branch into sync with the upstream repository, without losing your local changes:
        ```
        git merge upstream/master
        ```

        If your local branch didn't have any unique commits, git will instead perform a "fast-forward".  
        If there is a conflict and git couldn't merge, if you don't care for any modification in the local repo of your fork, you may need to git stash, then git pull, and repeat step 8.

        To repeat the sync of a fork in the future, you will have only to change to your local project (step 2) and start from step 6.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Git Commands Cheat Sheet

        From [https://github.com/ashash3/Cookies-and-Code-Git/wiki/Git-Commands-Cheat-Sheet](https://github.com/ashash3/Cookies-and-Code-Git/wiki/Git-Commands-Cheat-Sheet).

        **Basic Commands**   

        - `git status`: check the status of your files
        - `git add`: multipurpose command; track files, stage files, and mark merge conflicted files as resolved
        - `git diff`: compare working directory to staging area
        - `git diff --cached`: compare staged changes to last commit
        - `git commit –m "message"`: commit everything in your staging area, uses inline commit message
        - `git commit –a –m "message"`: automatically stage every currently tracked file and commits them (to skip “git add” command)
        - `git rm [filename]`: untrack the file and remove it from your working directory
        - `git rm --cached [filename]`: untrack the file, but keeps it in your working directory - useful if you forgot to include certain files in your .gitignore
        - `git mv [orig_name] [new_name]`: change the file's name
        - `git log` show the commit history in reverse chronological order (i.e. most recent first)
        "Undoing Things" Commands
        - `git commit --amend`: overrides your most recent commit - i.e. it "undoes" your most recent with what's currently in your staging area
        - `git reset HEAD [filename]`: allows you to unstage a particular file; this file returns back to the modified state
        - `git checkout -- [filename]`: allows you to discard any changes you've made to the file since the last commit Note: use this command carefully - the discarded changes cannot be recovered

        **Remote Repository Commands**

        - `git pull [remote-name] [branch-name]`: automatically fetch data from the remote server (typically called "origin") and attempts to merge it into the code you're working on; branch-name is typically "master" if you haven't created your own branch
        - `git push [remote-name] [branch-name]`: push your code from the branch you're on (typically "master" if you haven't created your own branch) upstream to the remote server (typically called "origin")

        **Merging and Branching Commands**

        - `git merge [branch-name]`: merge the specified branch with the current working directory
        - `git branch`: view all available branches
        - `git branch [branch-name]`: create a new branch
        - `git checkout [branch-name]`: set current working directory to branch-name
        - `git checkout -b [branch-name]`: create a new branch and set current working directory to it
        - `git merge [branch-name]`: merge branch-name into the current branch
        - `git branch -d [branch-name]`: delete the specified branch

        **Changing to Previous Commits Commands**

        - `git revert <prev_commit>`: create a new commit with a reverse patch that cancels out everything after that previous commit
        - `git checkout -b <branchname> <prev_commit>`: return to a previous commit and create a branch using it
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Further reading

        There is a lot of good materials on how to use Git for version control, here are a few:   

        - [Getting started with GitHub](https://docs.github.com/en/github/getting-started-with-github)  
        - [Git guide](http://rogerdudler.github.io/git-guide/)  
        - [Revision control software](http://nbviewer.ipython.org/urls/raw.github.com/jrjohansson/scientific-python-lectures/master/Lecture-7-Revision-Control-Software.ipynb)
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Video lectures on the Internet

         - [Git and GitHub for Beginners - Crash Course](https://youtu.be/RGOj5yH7evk)  
        """
    )
    return


@app.cell
def _():
    ## My workflow
    return


@app.cell
def _(BMC, DATA, GitHub, cd, mnt):
    cd /mnt/DATA/GitHub/BMC
    return


app._unparsable_cell(
    r"""
    !git pull
    """,
    name="_"
)


app._unparsable_cell(
    r"""
    !git add .
    """,
    name="_"
)


app._unparsable_cell(
    r"""
    !git commit -a -m \"comments\"
    """,
    name="_"
)


app._unparsable_cell(
    r"""
    !git push -u origin master
    """,
    name="_"
)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## References

         - [GitHub](https://github.com/)  
         - [Version control](https://en.wikipedia.org/wiki/Version_control) @ Wikipedia
        """
    )
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
