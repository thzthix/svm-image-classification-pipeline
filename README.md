# HOG + PCA + SVM Image Classification Pipeline

## 프로젝트 개요

이 프로젝트는 Fashion-MNIST 스타일 이미지 분류 문제를 대상으로,  
HOG feature extraction, PCA dimensionality reduction, SVM classification을 사용해  
기존 노트북 실험 코드를 재구성한 classical ML 기반 이미지 분류 파이프라인입니다.

목표는 실험용 노트북 코드를 그대로 두는 대신,  
학습과 추론 흐름을 분리하고 재사용 가능한 형태로 정리하는 것이었습니다.
---
title: Svm Image Classification Api
emoji: 🐠
colorFrom: indigo
colorTo: gray
sdk: docker
pinned: false
---

Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference

이 프로젝트는 Fashion-MNIST 이미지 분류 경진대회를 기반으로,  
HOG feature engineering, PCA dimensionality reduction, SVM classification을 활용하여 구현되었습니다.

기존 노트북 기반 코드를 다음 구조로 재구성했습니다:
- inference pipeline
- HOG/PCA 기반 feature space
- similarity search 시스템

## Environment

- Python 3.11
- `pip install -r requirements.txt`
