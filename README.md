# Semantic Characteristics of Schizophrenic Speech
This is the code for the paper: Semantic Characteristics of Schizophrenic Speech, which was published at CLPsych at NAACL 2019.
This was done as part of Vered Zilberstein's M.Sc research thesis in Computer Science at Tel Aviv University.


  * [Experiments - Transcribed Speech](#experiments---transcribed-speech)
  * [Experiments - Social Media](#experiments---social-media)
  * [Publications](#publications)
  * [Abstract](#Abstract)
  * [Video](#Video)
  * [Citation](#citation)
  * [License](#license)


## Experiments - Transcribed Speech
A much more detailed explanation is available in the paper.  

#### Input: 
Transcribed **speech** of schizophrenia inpatients who speak Hebrew.  
#### Experiment 1:
Measures derailment (topic mutation) in speech over time using word semantics.  
#### Experiment 2:
Measures incoherence by examining differences in usage of adjectives and adverbs (word modifiers) to describe content words.  
#### Classification: 
Takes scores calculated from both experiments and use them in a supervised classification task.  

### Results
#### Experiment 1
When using all words, we could not detect a significant difference between patients and controls.  
However, when using content words only, patients scored lower on derailment than the controls, for all window widths k, suggesting that focusing only on content words is the more robust approach for calculating derailment.

#### Experiment 2:
Controls have significantly higher scores for both modifier types, indicating a higher agreement on modifiers by the controls and external writers.

#### Classification:
Train several classifiers to distinguish between controls and patients. represent participants with the characteristics we compute in the two experiments. Random Forest achieves the highest accuracy of ~81%.

### High Level Design and Flow

This describes the design and flow of data into both experiments:  
![Semantic Characteristics of Schizophrenic Speech Design](./Semantic%20Characteristics%20of%20Schizophrenic%20Speech%20Design.png)

## Experiments - Social Media
We run the Experiment 1 on **written text**, taken from two social-media corpora  
### RSDD: 
Reddit self-reported depression diagnosis dataset - an existing dataset, as a source for users who are self-reported diagnosed with depression as well as a controls group. 
### TSSD: 
Twitter self-reported Schizophrenia diagnoses - our own collection of users who are self-reported diagnosed with schizophrenia as well as a controls group.  
Relevant code is available [here](https://github.com/Verose/Schizo_DS).

### Results
#### RSDD:
There is no significant difference between diagnosed users and controls.  
This result is expected, because depression patients do not tend to show derailment symptoms.

#### TSSD:
There is no significant difference between the groups.  
We suspect these results arise from Twitter being a medium for short, less-informative posts. It is also possible that written text is not as spontaneous as free speech, giving users time to organize their thoughts and edit multiple times before posting.

### High Level Design and Flow

![Speech Social Media Design](./Social%20Media%20Experiments.png)


## Publications

[This paper](https://www.aclweb.org/anthology/W19-3010.pdf) was presented at CLPsych at NAACL 2019.

## Abstract

Natural language processing tools are used to automatically detect disturbances in transcribed speech of
schizophrenia inpatients who speak Hebrew. We measure topic mutation over time and show that controls
maintain more cohesive speech than inpatients. We also examine differences in how inpatients and controls
use adjectives and adverbs to describe content words and show that the ones used by controls are more
common than those of inpatients. We provide experimental results and show their potential for automatically
detecting schizophrenia in patients by means only of their speech patterns. We then explore our findings
on publicly published written text taken from social media, which does not show a significant difference in
keeping cohesive discourse.

## Video

See the presentation of this paper at AI Week 2019:  
[![Semantic Characteristics of Schizophrenic Speech](https://img.youtube.com/vi/vVbP8wM1KxA/1.jpg)](https://www.youtube.com/watch?v=vVbP8wM1KxA)

## Citation

If you would like to cite this research, we would appreciate the following citation:

```console
@inproceedings{bar-etal-2019-semantic,
    title = "Semantic Characteristics of Schizophrenic Speech",
    author = "Bar, Kfir  and
      Zilberstein, Vered  and
      Ziv, Ido  and
      Baram, Heli  and
      Dershowitz, Nachum  and
      Itzikowitz, Samuel  and
      Vadim Harel, Eiran",
    booktitle = "Proceedings of the Sixth Workshop on Computational Linguistics and Clinical Psychology",
    month = jun,
    year = "2019",
    address = "Minneapolis, Minnesota",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/W19-3010",
    doi = "10.18653/v1/W19-3010",
    pages = "84--93",
    abstract = "Natural language processing tools are used to automatically detect disturbances in transcribed speech of schizophrenia inpatients who speak Hebrew. We measure topic mutation over time and show that controls maintain more cohesive speech than inpatients. We also examine differences in how inpatients and controls use adjectives and adverbs to describe content words and show that the ones used by controls are more common than the those of inpatients. We provide experimental results and show their potential for automatically detecting schizophrenia in patients by means only of their speech patterns.",
}
```

## License

This software is released under the [Apache License, Version 2.0](https://github.com/Verose/NLP_CLPSYCH/blob/master/LICENSE).
