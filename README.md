# Cypher-challenge-2025
Challenge on machine learning for turbulent combustion modeling organised for the cypher meetings in 2025
<p align="center">
  <img src="images/Challenge_overview_pic.png" alt="Challenge Overview" width="90%">
</p>

## Overview
Combustion systems play a vital role in transportation, energy production, and residential heating. Nonetheless, current combustion systems heavily rely on fossil fuels, which burning process is now acknowledged for being a source of greenhouse gases and consequently responsible for climate change [1 IPCC].
While dismantling completely our reliance on combustion is unfeasible for the so-called 'Hard to abate industries' [2 cerca su paper di parent], existing combustion processes can significantly reduce their CO2 emissions leveraging sustainable fuels such as hydrogen, ammonia, and methane [3 IDK what to cite]. A correct understanding of the physics of these fuels can help in the design of reliable and efficient systems that burn the aforementioned energy sources. One of the challenges in modeling these physical phenomena is the complex interaction between turbulence and chemistry in industrial systems.

Large eddy simulations (LES) stand as a powerful computational approach for studying turbulent combustion. LES directly resolves the large, energy-containing motions in the flow and filters out the smaller scales. Advances in high-performance computing and efficient algorithms have made this method increasingly accessible for practical use, but modeling the so-called sub-filter phenomena that arise at the unresolved smaller scales is a very challenging task. Part of the current and past efforts from the scientific community developed physics-based methods―such as transport equations―to link sub-filter quantities to the resolved field.

Machine learning (ML) is an interesting tool in improving turbulent combustion closure models for LES. Typical ML algorithms, such as Neural Networks, can analyze large datasets from direct numerical simulations (DNS), which provide highly accurate representations of turbulence and chemistry interactions. Recent studies highlight the potential of machine learning algorithms to develop closure models that outperform traditional physics-based approaches [cite Attili, Ludovico, etc].

This challenge strives to provide a benchmark for model assessment in the context of ML applied to turbulent combustion modeling. This first effort to provide a standard dataset and evaluation metric can be leveraged to foster consistency, facilitate model comparison, and accelerate progress in developing robust, generalizable data-driven closure models for reactive flows. The public nature of the challenge, combined with its implementation on the Codabench [cite] platform, ensures broad accessibility and promotes widespread dissemination within the research community.

## Scope of the challenge
This challenge strives to provide a benchmark for model assessment in the context of ML applied to turbulent combustion modeling. This first effort to provide a standard dataset and evaluation metric can be leveraged to foster consistency, facilitate model comparison, and accelerate progress in developing robust, generalizable data-driven closure models for reactive flows. The public nature of the challenge, combined with its implementation on the Codabench [cite] platform, ensures broad accessibility and promotes widespread dissemination within the research community.

Participants will be asked to submit a python code that defines a machine learning model, which will be trained after submission with the available data. All the models submitted will be evaluated on out-of-sample data. Tests in this contest will only be performed _a priori_ on the DNS dataset. As future research perspectives aim at coupling these models with Computational Fluid Dynamics (CFD) codes for _a posteriori_ validation, we apply penalisation accounting for the models' inference time. 

