# DLEVS: Deep Learning Enhanced Virtual Sensing for Aquaculture

**Paper:** [Deep Learning Enhanced Virtual Sensing: A Cost-Effective Approach for Water Quality Monitoring in Aquaculture](https://ieeexplore.ieee.org/abstract/document/11016948)

> Efficient monitoring of water quality parameters is crucial for the successful management of aquaculture systems by ensuring the health and growth of aquatic life. Traditional sensor-based systems are often cost-prohibitive, limiting their adoption, particularly in resource-constrained settings. In this research, we introduce a cost-effective method leveraging virtual sensing to estimate dissolved oxygen (DO) levels using solubility and interdependency equations. Furthermore, we propose deep learning-enhanced virtual sensing (DLEVS) employing deep recurrent neural networks, including long short-term memory, gated recurrent unit (GRU), their bidirectional variants, and attention mechanisms, to accurately estimate DO. With extensive hyperparameter tuning, we have identified that the attention-based GRU (A-GRU) model is optimal, achieving significant reductions in RMSE from 0.3802 to 0.1534 for ADAK dataset and from 0.7505 to 0.0848 for MAC dataset in comparison with conventional virtual sensing methods. Our findings demonstrate that the DLEVS mechanism delivers precise DO estimation with substantially reduced costs, presenting an appealing solution for aquaculture water quality monitoring.

This repository provides the full implementation of the DLEVS framework, a deep learning-based virtual sensing system designed to predict water quality parameters in aquaculture using historical sensor data.

## 📁 Repository Structure

DLEVS/
├── data/                   # Water quality datasets (ADAK, MAC)
├── notebooks/              # Jupyter notebooks for model training & evaluation
├── src/                    # Source code for models and preprocessing
├── requirements.txt        # Required Python packages
└── README.md               # Project documentation

## 🧠 Project Summary

DLEVS reduces dependency on physical sensors by using historical multivariate time-series data to predict current water quality parameters. It employs several deep learning architectures, including:

- LSTM
- GRU
- BiLSTM
- BiGRU
- CNN-LSTM hybrids

These models are evaluated using RMSE, MAE, and R² to assess prediction accuracy.

## 📊 Datasets

Included datasets:

- `WBG_ADAK.csv`, `data_adak.csv`: Sensor data from ADAK site
- `WBG_MAC.csv`, `data_mac.csv`: Sensor data from MAC site

Please refer to the `data/` folder for more details.

## 🚀 Setup & Usage

### 1. Clone the repository

git clone https://github.com/rasheedhaq/DLEVS.git
cd DLEVS

### 2. Create a virtual environment and install dependencies

python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt

### 3. Run experiments using Jupyter notebooks

Then open any notebook under the `notebooks/` folder to train and evaluate models:
- `All_Models_Order2.ipynb`
- `Wiess_ADAK.ipynb`
- `Wiess_MAC.ipynb`

---

## 📈 Results

Performance is assessed using:
- Root Mean Square Error (RMSE)
- Mean Absolute Error (MAE)
- R² Score

Each model’s output and plots are embedded in the respective notebooks.

## 📬 Contact

For queries or collaboration opportunities:

**Rasheed Abdul Haq K. P.**  
📧 [rasheedabdulhaq@gmail.com](mailto:rasheedabdulhaq@gmail.com)  
🔗 [https://github.com/rasheedhaq/DLEVS](https://github.com/rasheedhaq/DLEVS)
