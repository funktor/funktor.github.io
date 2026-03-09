---
layout: post
title:  "Distributed Training of a Deep Recommender System"
date:   2026-03-09 18:50:11 +0530
categories: ml
---
Designing a recommender system is not a trivial problem to solve and big tech companies have invested hundreds of millions into building the best recommender systems platform. Designing a production quality recommender system requires attention to several aspects few of which are highlighted below:<br/><br/>
1. 
While I will not go through the details of designing a recommender system in this post and I would like to keep that for another post in the future. In this post I would like to walkthrough the steps required to train a deep recommender system (a recommender system implemented using deep neural networks) in a distributed environment i.e. where we have a cluster of nodes/pods each with a limited memory and limited number of CPUs/GPUs. To keep the post short, I am assuming that the distributed environment is already setup in the cloud such as in GCP.
