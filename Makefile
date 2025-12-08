.PHONY: all featurize index train app

all: featurize index train

featurize:
	python src/features/build_text.py
	python src/features/build_txn.py
	python src/features/build_graph.py
	python src/features/fuse.py

index:
	python src/models/ann_index.py

train:
	python src/models/train_reranker.py
	python src/eval/report_metrics.py

app:
	streamlit run app/app.py
