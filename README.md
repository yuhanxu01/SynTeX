# SynTeX: Efficient LaTeX OCR with Synthetic Pretraining

## Overview

SynTeX is a data-efficient LaTeX OCR system that converts images of scientific documents into editable LaTeX code. Unlike existing methods requiring large-scale real paired datasets, SynTeX introduces a novel synthetic pretraining approach by randomly pairing grammatical Wikipedia text with LaTeX formulas, requiring no real LaTeX sources and only 400 fine-tuning samples.

## Method

SynTeX employs a two-stage training pipeline using Swin Transformer encoder with GPT2 decoder. The system first pretrains on 120k synthetic pages combining Wikipedia text and 2M LaTeX formulas, then adapts to real documents using 400 manually collected samples across English and Chinese, printed and handwritten content.


**Status**: Under review. Code, models, and datasets will be released.
