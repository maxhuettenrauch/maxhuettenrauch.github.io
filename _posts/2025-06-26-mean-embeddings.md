---
layout: distill
title: Mean Embeddings
date: 2025-06-26
description: or how we accidentally reinvented deep sets and forgot about graph neural networks 
tags: swarms gnns
giscus_comments: true

authors:
  - name: Maximilian Hüttenrauch

bibliography: 2025-06-26-mean-embeddings.bib
#categories: sample-posts
#related_posts: false
---

We proposed the idea of mean embeddings in my first JMLR paper, Deep Reinforcement Learning for Swarm Systems <d-cite key="huettenrauch2019deep"></d-cite>.
Instead of tweaking a learning algorithm, we decided to take a step back and ask ourselves how to actually represent information in a multi-agent task that provided a certain structure.
The standard approach during this time was to simply concatenate all information and then feed it through a network.
While this approach worked for a handful of agents, it lacked the scalability to system sizes of tens or even hundreds of agents.

Instead, we were looking for a more compact representation that was permutation invariant to the ordering and of a fixed size, independent of the number of agents.
In the context of swarms, where agents are assumed to be homogeneous and therefore interchangeable, we treated the observation items an agent has on neighboring agents as independent samples from a probability distribution. 
We then calculated an empirical estimate of the expected feature map (the mean embedding) to represent the distribution of observations as an element in a reproducing kernel Hilbert space.
This feature map provided an effective and well working representation in the case of a fully connected graph<d-footnote>Things obviously became more complicated once we limited the agents' communication radius, i.e., the maximum distance an agent could observe another agent, since mean embeddings only propagate information within a single hop.</d-footnote>.

In this post, I will quickly reintroduce the mean embedding and connect it to modern graph neural networks.
An implementation using pytorch can be found [here](https://github.com/maxhuettenrauch/swarm-rl).

## Observation Model 
Before we start, let's recap some definitions from <d-cite key="huettenrauch2019deep"></d-cite>.
In order to describe the observation model used for the agents, we use an interaction graph representation of the swarm.
This graph is given by nodes $V=\{v_1, v_2, \dots, v_N \}$ corresponding to the agents in the swarm and an edge set $E \subset V \times V$, which we assume contains unordered pairs of the form $\{v_i, v_j\}$ indicating that agents $i$ and $j$ are neighbors.
The interaction graph is denoted as $\mathcal{G} = (V, E)$.
The information agent $i$ observes about agent $j$ is denoted as $o^{i, j} = f(s^i, s^j)$, containing local geometric features as a function of the true states $s_i$ and $s_j$ of agent $i$ and agent $j$.
The observation $o^{i, j}$ is available for agent $i$ only if $$j \in \mathcal{N}_\mathcal{G}(i)$$ where $$\mathcal{N}_\mathcal{G}(i) = \{j \mid \{v_i, v_j\} \in E\}$$ is the neighborhood of agent $i$. 
The full information agent $i$ receives from all neighbors is given by the set $$O^i = \left\{o^{i, j} \mid j \in \mathcal{N}_\mathcal{G}(i) \right\}$$.
Additionally, each agent is able to observe local properties $o^i_\text{loc}$ such as its velocity.
The data we work with in this learning setup is decoupled from a global coordinate system.

## Mean Embeddings
As mentioned before, simply concatenating the items in $O^i$ and $o^i_\text{loc}$ has various drawbacks as it ignores the permutation invariance inherent to a homogeneous agent network.
Furthermore, it grows linearly with the number of agents in the swarm and is therefore limited to a fixed number of neighbors when used in combination with neural network policies.
A simple way to achieve permutation invariance of the elements of $O^i$ as well as flexibility to the size of $O^i$ is to use a mean feature embedding, i.e.,  

$$ 
\hat{\mu}_{O^i} = \frac{1}{|O^i|} \sum_{o^{i,j} \in O^i} \phi(o^{i, j}), 
$$

where $\phi$ defines the feature space of the mean embedding, realized by a standard feed forward neural network.
This embedding can then further be processed by the policy neural network $\pi$.
The resulting policy model 

$$
\pi(O^i, o^i_\text{loc}) = \rho(o^i_\text{loc}, \hat{\mu}_{O^i})
$$

is an instance of the invariant model proposed in <d-cite key="zaheer2017deep"></d-cite>, an approach to design models for machine learning tasks defined on sets<d-footnote>When we submitted this work for review, we were unaware of the DeepSets approach, but, luckily, one of the reviewers made us aware of it.</d-footnote>.

## Graph-based Mean Embeddings
Earlier, we already defined the swarm as a graph where the agents are the nodes and information can flow between them along edges. 
More precisely, the local features $o^i_\text{loc}$ describe the node features and $o^{i, j}$ the edge features in a directed graph.
These features are usually denoted as $x_i$ and $e_{i, j}$.
Using graph terminology, the action for agent $i$ is determined as a function

$$
\pi(\mathcal{G}, i) = \rho(x_i, \bigoplus_{j \in \mathcal{N}_\mathcal{G}(i)} \phi(e_{i, j}))
$$

where choosing $\bigoplus$ to be the mean results exactly in the mean embedding policy defined in the previous section.

## Message Passing Networks for Swarms
If we assume certain communication abilities between agents<d-footnote>In the paper, we assumed that agents can communicate the size of their neighborhood in the case of a limited communication radius. But why stop at a single feature?</d-footnote>, we can go one step further and include node features into the processing chain.
This becomes especially useful in case the graph is not complete.
For example, if communication is only possible in a certain range, or if the size of the swarm becomes too large and neighborhood sizes need to be restricted for the computations to remain tractable.
The process of obtaining an action is split up into a message passing phase and a readout phase <d-cite key="pmlr-v70-gilmer17a"></d-cite>.
In the message passing phase, a latent node feature 

$$
x^{k+1}_i = \rho(x^k_i, \bigoplus_{j \in \mathcal{N}_\mathcal{G}(i)} \phi(x^k_i, x^k_j, e_{i, j}))
$$

is updated based on the latent node and edge features using learned linear embeddings $x^0_v = M_x x_v$.
After $K$ steps, the action is determined as a function $\pi(x^K_i)$ where information has flown over up to $K$ hops.
