# Data provenance and definitions

The public [IBM Telco benchmark](https://github.com/IBM/telco-customer-churn-on-icp4d) contains 7,043 customer rows and 21 columns. download_data.py fetches data/Telco-Customer-Churn.csv from that repository; the observed byte fingerprint is stored in reports/run.json.

customerID is excluded from inputs. Churn=Yes is class 1, No is class 0. TotalCharges is converted to numeric; blank values are imputed from the training median. tenure is months; MonthlyCharges and TotalCharges use the benchmark's billing units. Categorical values are one-hot encoded and unseen categories are ignored at inference.

Downloaded data is not re-licensed or redistributed by this repository; consult IBM's source for terms and provenance. The original course notebook used a personal file; this maintenance run uses the publicly reproducible IBM mirror and records its fingerprint.

Methods: [scikit-learn pipelines](https://scikit-learn.org/stable/modules/compose.html), [pyGAM](https://pygam.readthedocs.io/en/latest/).
