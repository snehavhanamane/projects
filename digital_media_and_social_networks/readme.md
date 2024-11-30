# Analysing the Spreading of Memes on Social Media

This project explores the dynamics of meme propagation on social media networks using the **Reddit Hyperlink Network**. The analysis applies community detection algorithms, epidemic spreading models, and network centrality metrics.

## Features
- Dataset: Cross-links between Reddit subreddits over 40 months.
- Analysis Techniques:
  - SI (Susceptible-Infected) epidemic spreading model.
  - Centrality measures: Eigenvector and Betweenness.
  - Community detection using Louvain modularity optimization.
- Tools: Python, NetworkX, EoN, Gephi.

## Requirements
- Python 3.8+
- Libraries listed in `requirements.txt`

## Usage
1. Clone the repository:
   ```bash
   git clone https://github.com/username/meme-spreading-analysis.git


**Install dependencies:**
pip install -r requirements.txt

**Run the Jupyter notebook:**
jupyter notebook digital_media_and_social_networks.ipynb 


**Project Structure**
Dmsn_project_report.pdf: Detailed project report.
digital_media_and_social_networks.ipynb: Analysis notebook.
data/: Contains the dataset (not included due to size; link provided below).

**Dataset**
Reddit Hyperlink Network (137,821 cross-links between 35,766 subreddits).

**Results**
Community Detection: Insights into the role of network design in meme propagation.
SI Model Analysis: Visualized meme spread based on network centrality.
Node Removal Impact: Effects of removing key nodes on meme spread