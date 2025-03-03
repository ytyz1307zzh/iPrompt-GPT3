<p align="center">  <img src="https://microsoft.github.io/aug-models/embgam_gif.gif" width="18%"> 
<img align="center" width=40% src="https://csinva.io/imodelsX/imodelsx_logo.svg?sanitize=True&kill_cache=1"> </img>	<img src="https://microsoft.github.io/aug-models/embgam_gif.gif" width="18%"></p>

<p align="center">Library to explain <i>a dataset</i> in natural language. 
</p>
<p align="center">
  <a href="https://github.com/csinva/imodelsX/tree/master/demo_notebooks">📖 demo notebooks</a>
</p>
<p align="center">
  <img src="https://img.shields.io/badge/license-mit-blue.svg">
  <img src="https://img.shields.io/badge/python-3.9+-blue">
  <img src="https://img.shields.io/pypi/v/imodelsx?color=green">  
</p>  

| Model                       | Reference                                                    | Output  | Description                                                  |
| :-------------------------- | ------------------------------------------------------------ | ------- | ------------------------------------------------------------ |
| iPrompt            | [📖](https://github.com/csinva/imodelsX/blob/master/demo_notebooks/iprompt.ipynb), [🗂️](http://csinva.io/imodelsX/iprompt/api.html#imodelsx.iprompt.api.explain_dataset_iprompt), [🔗](https://github.com/csinva/interpretable-autoprompting), [📄](https://arxiv.org/abs/2210.01848) | Explanation | Generates a prompt that<br/>explains patterns in data (*Official*) |
| D3            | [📖](https://github.com/csinva/imodelsX/blob/master/demo_notebooks/d3.ipynb), [🗂️](http://csinva.io/imodelsX/d3/d3.html#imodelsx.d3.d3.explain_dataset_d3), [🔗](https://github.com/ruiqi-zhong/DescribeDistributionalDifferences), [📄](https://arxiv.org/abs/2201.12323) | Explanation | Explain the difference between two distributions |
| AutoPrompt            |  ㅤㅤ[🗂️](), [🔗](https://github.com/ucinlp/autoprompt), [📄](https://arxiv.org/abs/2010.15980) | Explanation | Find a natural-language prompt<br/>using input-gradients (⌛ In progress)|
| Aug-GAM            | [📖](https://github.com/csinva/imodelsX/blob/master/demo_notebooks/augmodels.ipynb), [🗂️](https://csinva.io/imodelsX/auggam/auggam.html), [🔗](https://github.com/microsoft/aug-models), [📄](https://arxiv.org/abs/2209.11799) | Linear model | Fit better linear model using an LLM<br/>to extract embeddings (*Official*) |
| Aug-Tree            | [📖](https://github.com/csinva/imodelsX/blob/master/demo_notebooks/augmodels.ipynb), [🗂️](https://csinva.io/imodelsX/augtree/augtree.html), [🔗](https://github.com/microsoft/aug-models), [📄](https://arxiv.org/abs/2209.11799) | Decision tree | Fit better decision tree using an LLM<br/>to expand features (*Official*) |
| SASC            |  ㅤㅤ[🗂️](https://csinva.io/imodelsX/sasc/api.html), [🔗](https://github.com/microsoft/automated-explanations), [📄](https://arxiv.org/abs/2305.09863) | Explanation | Explain a black-box text module<br/>using an LLM (*Official*) |
| Linear Finetune  | [📖](https://github.com/csinva/imodelsX/blob/master/demo_notebooks/linearfinetune.ipynb), [🗂️](https://csinva.io/imodelsX/linear_finetune.html) | Black-box model | Finetune a single linear layer<br/>on top of LLM embeddings |

<p align="center">
<a href="https://github.com/csinva/imodelsX/tree/master/demo_notebooks">📖</a>Demo notebooks &emsp; <a href="https://csinva.io/imodelsX/">🗂️</a> Doc &emsp; 🔗 Reference code &emsp; 📄 Research paper
</br>
⌛ We plan to support other interpretable algorithms like <a href="https://arxiv.org/abs/2205.12548">RLPrompt</a>, <a href="https://arxiv.org/abs/2007.04612">CBMs</a>, and <a href="https://arxiv.org/abs/2004.00221">NBDT</a>. If you want to contribute an algorithm, feel free to open a PR 😄
</p>


# Quickstart
**Installation**: `pip install imodelsx` (or, for more control, clone and install from source)

**Demos**: see the [demo notebooks](https://github.com/csinva/imodelsX/tree/master/demo_notebooks)

### iPrompt

```python
from imodelsx import explain_dataset_iprompt, get_add_two_numbers_dataset

# get a simple dataset of adding two numbers
input_strings, output_strings = get_add_two_numbers_dataset(num_examples=100)
for i in range(5):
    print(repr(input_strings[i]), repr(output_strings[i]))

# explain the relationship between the inputs and outputs
# with a natural-language prompt string
prompts, metadata = explain_dataset_iprompt(
    input_strings=input_strings,
    output_strings=output_strings,
    checkpoint='EleutherAI/gpt-j-6B', # which language model to use
    num_learned_tokens=3, # how long of a prompt to learn
    n_shots=3, # shots per example
    n_epochs=15, # how many epochs to search
    verbose=0, # how much to print
    llm_float16=True, # whether to load the model in float_16
)
--------
prompts is a list of found natural-language prompt strings
```

### D3 (DescribeDistributionalDifferences)

```python
from imodelsx import explain_dataset_d3
hypotheses, hypothesis_scores = explain_dataset_d3(
    pos=positive_samples, # List[str] of positive examples
    neg=negative_samples, # another List[str]
    num_steps=100,
    num_folds=2,
    batch_size=64,
)
```

### Aug-models
Use these just a like a scikit-learn model. During training, they fit better features via LLMs, but at test-time they are extremely fast and completely transparent.

```python
from imodelsx import AugGAMClassifier, AugTreeClassifier, AugGAMRegressor, AugTreeRegressor
import datasets
import numpy as np

# set up data
dset = datasets.load_dataset('rotten_tomatoes')['train']
dset = dset.select(np.random.choice(len(dset), size=300, replace=False))
dset_val = datasets.load_dataset('rotten_tomatoes')['validation']
dset_val = dset_val.select(np.random.choice(len(dset_val), size=300, replace=False))

# fit model
m = AugGAMClassifier(
    checkpoint='textattack/distilbert-base-uncased-rotten-tomatoes',
    ngrams=2, # use bigrams
)
m.fit(dset['text'], dset['label'])

# predict
preds = m.predict(dset_val['text'])
print('acc_val', np.mean(preds == dset_val['label']))

# interpret
print('Total ngram coefficients: ', len(m.coefs_dict_))
print('Most positive ngrams')
for k, v in sorted(m.coefs_dict_.items(), key=lambda item: item[1], reverse=True)[:8]:
    print('\t', k, round(v, 2))
print('Most negative ngrams')
for k, v in sorted(m.coefs_dict_.items(), key=lambda item: item[1])[:8]:
    print('\t', k, round(v, 2))
```

### Linear finetune
An easy-to-fit baseline that follows the same API.

```python
from imodelsx import LinearFinetuneClassifier
# fit a simple one-layer finetune
m = LinearFinetuneClassifier(
    checkpoint='distilbert-base-uncased',
)
m.fit(dset['text'], dset['label'])
preds = m.predict(dset_val['text'])
acc = (preds == dset_val['label']).mean()
print('validation acc', acc)
```

### SASC
Here, we explain a *module* rather than a dataset

```python
from imodelsx import explain_module_sasc
# a toy module that responds to the length of a string
mod = lambda str_list: np.array([len(s) for s in str_list])

# a toy dataset where the longest strings are animals
text_str_list = ["red", "blue", "x", "1", "2", "hippopotamus", "elephant", "rhinoceros"]
explanation_dict = explain_module_sasc(
    text_str_list,
    mod,
    ngrams=1,
)
```


# Related work
- imodels package (JOSS 2021 [github](https://github.com/csinva/imodels)) - interpretable ML package for concise, transparent, and accurate predictive modeling (sklearn-compatible).
- Adaptive wavelet distillation (NeurIPS 2021 [pdf](https://arxiv.org/abs/2107.09145), [github](https://github.com/Yu-Group/adaptive-wavelets)) - distilling a neural network into a concise wavelet model
- Transformation importance (ICLR 2020 workshop [pdf](https://arxiv.org/abs/2003.01926), [github](https://github.com/csinva/transformation-importance)) - using simple reparameterizations, allows for calculating disentangled importances to transformations of the input (e.g. assigning importances to different frequencies)
- Hierarchical interpretations (ICLR 2019 [pdf](https://openreview.net/pdf?id=SkEqro0ctQ), [github](https://github.com/csinva/hierarchical-dnn-interpretations)) - extends CD to CNNs / arbitrary DNNs, and aggregates explanations into a hierarchy
- Interpretation regularization (ICML 2020 [pdf](https://arxiv.org/abs/1909.13584), [github](https://github.com/laura-rieger/deep-explanation-penalization)) - penalizes CD / ACD scores during training to make models generalize better
- PDR interpretability framework (PNAS 2019 [pdf](https://arxiv.org/abs/1901.04592)) - an overarching framewwork for guiding and framing interpretable machine learning
