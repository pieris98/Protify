#!/bin/bash

echo "Running transfer learning case study with ESMC-600 on 2 sparse annotation tasks"

# 1. LoRA
python -m main --model_names ESMC-600 --data_names EC GO-BP --patience 3 --max_length 1024 --num_runs 3 --lora --seed 1184471526 > /dev/null 2>&1
zip -q -r ESMC600_LoRA.zip results plots logs
rm -rf results plots logs

# 2. LoRA + Hybrid Probe
python -m main --model_names ESMC-600 --data_names EC GO-BP --patience 3 --max_length 1024 --num_runs 3 --lora --hybrid_probe --seed 1184471526 > /dev/null 2>&1
zip -q -r ESMC600_LoRA_Hybrid.zip results plots logs
rm -rf results plots logs

# 3. Hybrid Probe (only)
python -m main --model_names ESMC-600 --data_names EC GO-BP --patience 3 --max_length 1024 --num_runs 3 --hybrid_probe --seed 1184471526 > /dev/null 2>&1
zip -q -r ESMC600_Hybrid.zip results plots logs
rm -rf results plots logs

# 4. Frozen / Base (No flags)
python -m main --model_names ESMC-600 --data_names EC GO-BP --patience 3 --max_length 1024 --num_runs 3 --seed 1184471526 > /dev/null 2>&1
zip -q -r ESMC600_LinearProbe.zip results plots logs
rm -rf results plots logs

# 5. Transformer Probe
python -m main --model_names ESMC-600 --data_names EC GO-BP --patience 3 --max_length 1024 --num_runs 3 --transformer_probe --seed 1184471526 > /dev/null 2>&1
zip -q -r ESMC600_TransformerProbe.zip results plots logs
rm -rf results plots logs

# 6. Full Finetuning
python -m main --model_names ESMC-600 --data_names EC GO-BP --patience 3 --max_length 1024 --num_runs 3 --full_finetuning --seed 1184471526 > /dev/null 2>&1
zip -q -r ESMC600_FullFinetuning.zip results plots logs
rm -rf results plots logs