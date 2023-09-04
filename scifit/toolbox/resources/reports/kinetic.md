---
title: {{title}}
author: {{author}}
numbersections: true
toc: true
urlcolor: blue
linkcolor: red
header-includes:
  - \usepackage{amsmath}
  - \usepackage{mathtools}
  - \usepackage{longtable}
  - \usepackage{booktabs}
  - \usepackage{siunitx}
  - \sisetup{tight-spacing=true,round-precision=4}
---
# Summary

## Information

This report has been automatically generated by the SciFit package.
It includes all mandatory information about the data, model and resolution.
See our [repository](https://github.com/jlandercy/scifit) for methodology details.

\clearpage

# Report

The solver uses the Activated State Model theory to represent kinetic reactions
with input data formatted in a matrix fashion: rate monoms, reaction rate polynoms and
reaction polynoms quotients. Kinetics ODE system are notably stiff, albeit conditionning
and numerical stability are taken into account, jitter may happen  for some specific
set of parameters or at large time scale. Always keep your critical sense when inspecting
figures and tables.

## Model

The model is based on the following kinetics:

{{equations}}

Which are described by the following stoechiometric coefficients matrix:

{{coefficients}}

With initial concentrations vector and steadiness conditions:

{{concentrations}}

And kinetic constants vector:

{{constants}}

Time domain for solver is defined as:

\begin{center}
\begin{tabular}{rS}
$t_{\min}$ & {{tmin}} \\
$t_{\max}$ & {{tmax}} \\
$\mathrm{d}t$ & {{dt}} \\
$n$ & {{n}}
\end{tabular}
\end{center}

## Solutions

Figure \ref{fig:solution} shows the time dependant solutions of the system dynamic.

![Solution of the ODE System\label{fig:solution}]({{solution}}){width=350px}

Where each substance rate is defined as follows:

{% if mode == "direct" %}
\begin{equation}
r_l^{\rightarrow} = \sum\limits_{i=1}^{i=n} \nu_{i,j} \cdot k_l^{\rightarrow} \cdot \prod\limits_{j=1}^{j=k} x_j^{|\nu_{l,j}^R|} \, , \quad \forall l \in \{1,\dots, n\}
\end{equation}
{% elif mode == "indirect" %}
\begin{equation}
r_l^{\leftarrow} = \sum\limits_{i=1}^{i=n} \nu_{i,j} \cdot k_l^{\leftarrow} \cdot \prod\limits_{j=1}^{j=k} x_j^{|\nu_{l,j}^P|} \, , \quad \forall l \in \{1,\dots, n\}
\end{equation}
{% elif mode == "equilibrium" %}
\begin{equation}
r_l^{\leftrightharpoons} = \sum\limits_{i=1}^{i=n} \nu_{i,j} \left( \cdot k_l^{\rightarrow} \cdot \prod\limits_{j=1}^{j=k} x_j^{|\nu_{l,j}^R|} - k_l^{\leftarrow} \cdot \prod\limits_{j=1}^{j=k} x_j^{|\nu_{l,j}^P|} \right) \, , \quad \forall l \in \{1,\dots, n\}
\end{equation}
{% endif %}


Figure \ref{fig:quotients} shows reaction quotients during the scenario.

![Reaction Quotients\label{fig:quotients}]({{quotients}}){width=250px}

Where each quotient is defined as follows:

\begin{equation}
Q_i = \prod\limits_{j=1}^{j=k} x_j^{\nu_{i,j}} \, , \quad \forall i \in \{1,\dots, n\}
\end{equation}


Figure \ref{fig:selectivities} shows reaction selectivities during the scenario.

![Reaction Selectivities\label{fig:selectivities}]({{selectivities}}){width=250px}


\clearpage

# Annexe

## Data
Table \ref{tab:data} presents a subset of solution and quotients.

\tiny
{{dataset}}\label{tab:data}
\normalsize