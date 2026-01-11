media_bias_news/
│
├── data/
│   ├── raw/
│   │   └── huffpost_links.json
│   ├── intermediate/
│   │   └── selected_links_50k.json
│   └── processed/
│       └── news_50k.csv
│
├── scraping/
│   └── fetch_articles.py
│
├── preprocessing/
│   ├── select_links.py
│   ├── clean_text.py
│   └── create_dataset.py
│
├── models/
│   └── distilbert_embeddings.py
│
├── analysis/
│   └── discourse_analysis.py
│
├── streamlit_app.py   (plus tard)
├── requirements.txt
└── README.md
