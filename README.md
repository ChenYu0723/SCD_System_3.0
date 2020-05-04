# ITS with OD-STGCN method.

Before you run this project, maybe you need metro raw data: metroData_filtered.csv (433MB).

If you don't have it, please download it. Then put it to "data/raw_data/metroData_filtered.csv".

## Code structure
```
.
├── README.md
├── __pycache__
├── code
│   ├── __init__.py
│   ├── __pycache__
│   ├── baselines.py
│   ├── demo.py
│   ├── gcn.py
│   ├── gru.py
│   ├── main.py
│   ├── plot_error.py
│   ├── reload_model.py
│   ├── result.txt
│   ├── tgcn.py
│   └── utils
├── data
│   ├── adj_matrix
│   ├── corr_matrix
│   ├── od_matrix
│   ├── raw_data
│   └── station_flow
├── model
│   ├── adj_1000
│   ├── demo
│   ├── gru_500
│   ├── od_1000
│   └── od_filtered
└── output
    ├── error_fig
    ├── error_fig_en
    └── flow_fig

```

## Paper
If you want to know the detail of this method, please see the paper from https://www.overleaf.com/read/nbnxpnykqmgg.

And if you want to edit it, please contact me.