<!-- Improved compatibility of back to top link: See: https://github.com/othneildrew/Best-README-Template/pull/73 -->
<a name="readme-top"></a>
<!--
*** Thanks for checking out the Best-README-Template. If you have a suggestion
*** that would make this better, please fork the repo and create a pull request
*** or simply open an issue with the tag "enhancement".
*** Don't forget to give the project a star!
*** Thanks again! Now go create something AMAZING! :D
-->



<!-- PROJECT SHIELDS -->
<!--
*** I'm using markdown "reference style" links for readability.
*** Reference links are enclosed in brackets [ ] instead of parentheses ( ).
*** See the bottom of this document for the declaration of the reference variables
*** for contributors-url, forks-url, etc. This is an optional, concise syntax you may use.
*** https://www.markdownguide.org/basic-syntax/#reference-style-links
-->



<!-- PROJECT LOGO -->
<br />
<div align="center">
  <a href="https://github.com/github_username/repo_name">
    <img src="images/Space Titanic.png" alt="Logo" width="512" height="512">
  </a>

<h3 align="center">Project Space Titanic</h3>

  <p align="center">
    A project trying to solve the Space Titanic Competition on Kaggle
  </p>
</div>







<!-- ABOUT THE PROJECT -->
## About The Project



The project is coded with pure python, and the provided files are supposed to be executed via Jupyter Notebook.
The Space Titanic competition is a bi-class classification task. Therefore, We are using a 9-12 layer MLP to complete this task.

<p align="right">(<a href="#readme-top">back to top</a>)</p>



### Built With

* Pytorch
* Numpy
* Pandas

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- GETTING STARTED -->
## Getting Started

If you are already a python user and have Jupyter Notebook installed, you could open this project and skip this part.
Both SpaceTitanicV0 and V1 should work fine, but there is a minor difference in data preprocessing (and it has a minor-to-none influence on the model).

### Prerequisites

This is an example of how to list things you need to use the software and how to install them.
* Install Jupyter Notebook via [Anaconda](https://www.anaconda.com/products/distribution)
* Access Anaconda Prompt (and activate environment)
    ```sh
      activate [Environment Name]
* Use pip for package installation
    ```sh
      pip install pytorch
      pip install pandas
      pip install numpy

<!-- USAGE EXAMPLES -->
## Usage

Open either SpaceTitanicV0 or V1, and execute all. You may change the structure of the Neural Net, activation function, optimizer, batch size, and learning rate if you want.
But we recommend using Adam (or any fast optimizer) because otherwise it may take much time on a personal computer.

To load saved models, you can edit torch.save(model, 'model/[model name]') and run the file with training skipped.

<!-- ROADMAP -->
## Roadmap

- [ ] SpaceTitanicV0 - the basic version
- [ ] SpaceTitanicV1 - last name feature is maintained in the training data
- [ ] Upcoming V2 - simplify the data preprocessing steps
- [ ] Variant Symmetry Cancellation - design for studying the symmetric in model weights (not included yet)


<p align="right">(<a href="#readme-top">back to top</a>)</p>






<!-- CONTACT -->
## Contact

Shaowen Wen - [shaowen0219@outlook.com](https://shaowen0219@outlook.com)

Project Link: [https://github.com/github_username/repo_name](https://github.com/github_username/repo_name)

<p align="right">(<a href="#readme-top">back to top</a>)</p>



