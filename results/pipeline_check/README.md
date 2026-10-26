# Pipeline check (not a thesis result)

`tiny_bert_from_scratch.*` comes from:

```bash
python scripts/train_classifier.py --smoke-test --smoke-sample 0 --epochs 3 --lr 1e-3 --batch-size 32 \
    --out results/pipeline_check/tiny_bert_from_scratch
```

A 2-layer BERT with random weights and a tokenizer trained on our own data, trained on CPU in under 4 minutes. It exists only to prove the week 9 code is wired correctly (chunk labels, class weights, training loop, document-level scoring), because pretrained models couldn't be downloaded on the machine where the code was written.

It learned: chunk-level validation AUROC rose to 0.98 and document-level test AUROC was 0.90. Results from the real pretrained model come from running `notebooks/colab_weeks8-9.ipynb`.
