# Cypher-challenge-2025
Challenge on machine learning for turbulent combustion modeling organised within the context of the CYPHER COST Action https://cypher.ulb.be/
<p align="center">
  <img src="images/Challenge_overview_pic.png" alt="Challenge Overview" width="90%">
</p>

## Overview
Combustion systems play a vital role in transportation, energy production, and residential heating. Nonetheless, current combustion systems heavily rely on fossil fuels, whose burning process is now acknowledged as a source of greenhouse gases and consequently responsible for climate change.
While completely dismantling our reliance on combustion is unfeasible for the so-called 'hard-to-abate industries' [ https://doi.org/10.1016/j.resconrec.2024.107796 ], existing combustion processes can significantly reduce their CO2 emissions. In this context, the CYPHER COST action is dedicated to advancing the understanding of Renewable Synthetic Fuels (RSFs) combustion, high-fidelity simulations, hybrid physics-based data-driven models, and self-updating digital twins. 

Regarding combustion simulations, the Large eddy simulations (LES) stand as a powerful approach for studying turbulent combustion. LES directly resolves the large, energy-containing motions in the flow and filters out the smaller scales. Advances in high-performance computing and efficient algorithms have made this method increasingly accessible for practical use, but modeling the so-called sub-filter phenomena that arise at the unresolved smaller scales is still a challenging task, in particular for RSFs. While significant past efforts from the scientific community have developed physics-based methods to link sub-filter quantities to the resolved LES field, nowadays, approaches based on high-fidelity data are becoming popular. Within this framework, machine learning (ML) is an interesting tool in improving turbulent combustion closure models for LES. Typical ML algorithms, such as Neural Networks, can analyze large datasets from direct numerical simulations (DNS), which provide highly accurate representations of turbulence and chemistry interactions. Recent studies highlight the potential of machine learning algorithms to develop closure models that outperform traditional physics-based approaches. 

## Scope of the challenge
This challenge aims to establish a benchmark for model assessment in the context of machine learning (ML) applied to turbulent combustion modeling. This first effort to provide a standard dataset and evaluation metric can be leveraged to foster consistency, facilitate model comparison, and accelerate progress in developing robust, generalizable data-driven closure models for reactive flows. The public nature of the challenge, combined with its implementation on the Codabench https://doi.org/10.1016/j.patter.2022.100543 platform, ensures broad accessibility and promotes widespread dissemination within the research community.

Participants will be asked to submit a Python code that defines a machine learning model, which will be trained after submission with the available data. All the models submitted will be evaluated on out-of-sample data. Tests in this contest will only be performed _a priori_ on the DNS dataset. As future research perspectives aim at coupling these models with Computational Fluid Dynamics (CFD) codes for _a posteriori_ validation, we apply penalisation accounting for the model's inference time. 

All the technical information can be found at https://cypher.ulb.be/data-challenge/




