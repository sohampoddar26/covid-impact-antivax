## Cov-Gen Method

1. First install all requirements listed in requirements.txt
2. Run ```python data_preparation.py ```
This will preprocess tweet and prepare data for flant5 instruction tuning.
3. Run ```python CovGen_lora_instruction_tuning.py ```
4. Run ```python label_matcher.py ``` . This will also show the performance metrics
